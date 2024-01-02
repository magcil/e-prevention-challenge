import os
import sys
import argparse
import json
from datetime import datetime

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_PATH)

import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
import pandas as pd
from tqdm import tqdm

from src.end_to_end_evaluation import get_model
import utils.parse as parser
from datasets.dataset import PatientDataset
from utils.util_funcs import svm_score, fill_predictions, apply_postprocessing_filter, calculate_roc_pr_auc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Submission json configuration file.")

    return parser.parse_args()


def submission_loop(track_id, patient_id, patient_config, train_dloader, val_dloader, test_dloader, model,
                    one_class_test, device):

    model = model.to(device)
    model.eval()
    train_losses, train_embeddings = [], []
    with torch.no_grad():
        # Loop over train and fit normal distribution / SVM
        for d in train_dloader:
            features, mask = d['features'], d['mask']
            features, mask = features.to(device), mask.to(device)

            reco_features, emb = model(features)
            reco_features = reco_features * mask

            emb = torch.flatten(emb, start_dim=1)
            loss = (reco_features - features)**2
            loss = np.mean(loss.cpu().numpy(), axis=(1, 2, 3))

            train_losses.append(loss)
            train_embeddings.append(emb.cpu().numpy())

        train_losses, train_embeddings = np.concatenate(train_losses), np.concatenate(train_embeddings)

        # Check if one_class_test is True
        if one_class_test:
            detector = OneClassSVM()
            scaler = StandardScaler()
            # Fit scaler & SVM
            scaler.fit(train_embeddings)
            train_embeddings = scaler.transform(train_embeddings)
            detector.fit(train_embeddings)

        else:
            mu, std = np.mean(train_losses), np.std(train_losses)

        # Loop over validation set / sanity check
        test_losses, test_embeddings = [], []
        # Other info
        splits, days, labels = [], [], []
        for d in val_dloader:
            features, mask = d['features'], d['mask']
            features, mask = features.to(device), mask.to(device)
            reco_features, emb = model(features)
            reco_features = reco_features * mask

            test_embeddings.append(torch.flatten(emb, start_dim=1).cpu().numpy())
            loss = (reco_features - features)**2
            loss = np.mean(loss.cpu().numpy(), axis=(1, 2, 3))
            test_losses.append(loss)

            splits.append(d['split'])
            days.append(d['day_index'].numpy())
            labels.append(d['label'].numpy())

        test_losses, splits, days, labels = np.concatenate(test_losses), np.concatenate(splits), np.concatenate(
            days), np.concatenate(labels)
        test_embeddings = np.concatenate(test_embeddings)

        df = pd.DataFrame({"split": splits, "day_index": days, "test_loss": test_losses, "label": labels})

        if one_class_test:
            test_embeddings = scaler.transform(test_embeddings)
            preds = detector.predict(test_embeddings)
            dist_from_hyperplane = detector.decision_function(test_embeddings)
            split_day = [split + "_day_" + str(day) for split, day in zip(splits, days)]

            df['split_day'] = split_day
            df['preds'] = preds
            df['dist_from_hp'] = dist_from_hyperplane

            anomaly_scores, split, days, labels = [], [], [], []

            # Aggregate results for each day
            for sp in df['split_day'].unique():
                filt_df = df[df['split_day'] == sp]
                score = svm_score(filt_df['preds'].to_numpy(), filt_df[filt_df['dist_from_hp'] < 0]['dist_from_hp'])
                anomaly_scores.append(score)

                split.append(filt_df['split'].iloc[0])
                days.append(filt_df['day_index'].iloc[0])
                labels.append(filt_df['label'].iloc[0])

            # Process results for each split
            res = pd.DataFrame({"split": split, "day_index": days, "label": labels, "anomaly_scores": anomaly_scores})
        else:
            res = df.groupby(by=["split", "day_index"]).mean()
            res['label'] = res['label'].astype(int)
            res['anomaly_scores'] = res['test_loss'].map(lambda x: 1 - norm.pdf(x, mu, std) / norm.pdf(mu, mu, std))
            res.reset_index(names=["split", "day_index"], inplace=True)

        aggr_scores, aggr_scores_filtered, aggr_labels = [], [], []
        for sp in res['split'].unique():
            filt_df = res[res['split'] == sp].sort_values(by="day_index")
            # Fill predictions
            csv_to_save = fill_predictions(track_id=track_id,
                                           patient_id=patient_id,
                                           anomaly_scores=filt_df['anomaly_scores'],
                                           split=sp,
                                           days=filt_df['day_index'])
            csv_to_save['anomaly_scores_filtered'] = apply_postprocessing_filter(
                csv_to_save['anomaly_scores'].to_numpy(), patient_config['filter'], patient_config['filter_size'])
            mode, num = sp.split("_")[0], int(sp.split("_")[1])

            # Save validation results
            csv_to_save[["day_index", "relapse", "anomaly_scores", "anomaly_scores_filtered"
                         ]].to_csv(os.path.join(parser.get_path(track=track_id, patient=patient_id, mode=mode, num=num),
                                                f"submissions_{datetime.today().date()}.csv"),
                                   index=False)

            aggr_scores.append(csv_to_save['anomaly_scores'].to_numpy())
            aggr_scores_filtered.append(csv_to_save['anomaly_scores_filtered'].to_numpy())
            aggr_labels.append(csv_to_save['relapse'].to_numpy())

        aggr_scores, aggr_scores_filtered, aggr_labels = np.concatenate(aggr_scores), np.concatenate(
            aggr_scores_filtered), np.concatenate(aggr_labels)

        # Calculate scores on validation
        scores = calculate_roc_pr_auc(aggr_scores, aggr_labels)
        scores_filtered = calculate_roc_pr_auc(aggr_scores_filtered, aggr_labels)

        # Loop over the test set
        test_losses, test_embeddings = [], []
        # Other info
        splits, days = [], []
        for d in test_dloader:
            features, mask = d['features'], d['mask']
            features, mask = features.to(device), mask.to(device)
            reco_features, emb = model(features)
            reco_features = reco_features * mask

            test_embeddings.append(torch.flatten(emb, start_dim=1).cpu().numpy())
            loss = (reco_features - features)**2
            loss = np.mean(loss.cpu().numpy(), axis=(1, 2, 3))
            test_losses.append(loss)

            splits.append(d['split'])
            days.append(d['day_index'].numpy())

        test_losses, splits, days = np.concatenate(test_losses), np.concatenate(splits), np.concatenate(days)
        test_embeddings = np.concatenate(test_embeddings)

        df = pd.DataFrame({"split": splits, "day_index": days, "test_loss": test_losses})

        if one_class_test:
            test_embeddings = scaler.transform(test_embeddings)
            preds = detector.predict(test_embeddings)
            dist_from_hyperplane = detector.decision_function(test_embeddings)
            split_day = [split + "_day_" + str(day) for split, day in zip(splits, days)]

            df['split_day'] = split_day
            df['preds'] = preds
            df['dist_from_hp'] = dist_from_hyperplane

            anomaly_scores, split, days = [], [], []

            # Aggregate results for each day
            for sp in df['split_day'].unique():
                filt_df = df[df['split_day'] == sp]
                score = svm_score(filt_df['preds'].to_numpy(), filt_df[filt_df['dist_from_hp'] < 0]['dist_from_hp'])
                anomaly_scores.append(score)

                split.append(filt_df['split'].iloc[0])
                days.append(filt_df['day_index'].iloc[0])

            # Process results for each split
            res = pd.DataFrame({"split": split, "day_index": days, "score": anomaly_scores})
        else:
            res = df.groupby(by=["split", "day_index"]).mean()
            res['score'] = res['test_loss'].map(lambda x: 1 - norm.pdf(x, mu, std) / norm.pdf(mu, mu, std))
            res.reset_index(names=["split", "day_index"], inplace=True)

        # Generate submissions
        for sp in res['split'].unique():
            filt_df = res[res['split'] == sp].sort_values(by="day_index")
            # Fill predictions
            csv_to_save = fill_predictions(track_id=track_id,
                                           patient_id=patient_id,
                                           anomaly_scores=filt_df['score'],
                                           split=sp,
                                           days=filt_df['day_index'])
            csv_to_save['anomaly_scores'] = apply_postprocessing_filter(csv_to_save['anomaly_scores'].to_numpy(),
                                                                        patient_config['filter'],
                                                                        patient_config['filter_size'])
            mode, num = sp.split("_")[0], int(sp.split("_")[1])

            # Save validation results
            csv_to_save.rename(columns={"anomaly_scores": "score"}, inplace=True)

            csv_to_save[["score", "day_index"
                         ]].to_csv(os.path.join(parser.get_path(track=track_id, patient=patient_id, mode=mode, num=num),
                                                f"submissions_{datetime.today().date()}.csv"),
                                   index=False)

    return {
        "ROC AUC": scores["ROC AUC"],
        "PR AUC": scores["PR AUC"],
        "ROC AUC (filter)": scores_filtered["ROC AUC"],
        "PR AUC (filter)": scores_filtered["PR AUC"]
    }


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r") as f:
        json_config = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {"patient_id": [], "ROC AUC": [], "PR AUC": [], "ROC AUC (filter)": [], "PR AUC (filter)": []}

    # Evaluate on each patient and generate submission
    for patient_id in tqdm(json_config["patients"]):
        # Get patient config
        patient_config = json_config["patients_config"][f"P{patient_id}"]
        patient_path = parser.get_path(track=json_config["track_id"], patient=patient_id)
        # Load model
        model = get_model(model_str=patient_config["model"],
                          window_size=json_config["window_size"],
                          num_layers=patient_config["num_layers"])
        model.load_state_dict(torch.load(os.path.join(PROJECT_PATH, patient_config["pt_file"])))

        train_dset = PatientDataset(track_id=json_config["track_id"],
                                    patient_id=patient_id,
                                    mode="train",
                                    window_size=json_config["window_size"],
                                    extension=json_config["file_format"],
                                    feature_mapping=json_config["feature_mapping"],
                                    from_path=True)
        val_dset = PatientDataset(track_id=json_config["track_id"],
                                  patient_id=patient_id,
                                  mode="val",
                                  window_size=json_config["window_size"],
                                  extension=json_config["file_format"],
                                  feature_mapping=json_config["feature_mapping"],
                                  from_path=True)

        test_dset = PatientDataset(track_id=json_config["track_id"],
                                   patient_id=patient_id,
                                   mode="test",
                                   window_size=json_config["window_size"],
                                   extension=json_config["file_format"],
                                   feature_mapping=json_config["feature_mapping"],
                                   from_path=True)

        train_dset._cal_statistics()
        val_dset.mean, val_dset.std = train_dset.mean, train_dset.std
        test_dset.mean, test_dset.std = train_dset.mean, train_dset.std

        train_dset._upsample_data(json_config["upsampling_size"])
        val_dset._upsample_data(json_config["prediction_upsampling"])
        test_dset._upsample_data(json_config["prediction_upsampling"])

        train_dloader = DataLoader(train_dset,
                                   batch_size=json_config["batch_size"],
                                   shuffle=False,
                                   num_workers=json_config["num_workers"])
        val_dloader = DataLoader(val_dset,
                                 batch_size=json_config["batch_size"],
                                 shuffle=False,
                                 num_workers=json_config["num_workers"])
        test_dloader = DataLoader(test_dset,
                                  batch_size=json_config["batch_size"],
                                  shuffle=False,
                                  num_workers=json_config["num_workers"])

        patient_results = submission_loop(track_id=json_config["track_id"],
                                          patient_id=patient_id,
                                          patient_config=patient_config,
                                          train_dloader=train_dloader,
                                          val_dloader=val_dloader,
                                          test_dloader=test_dloader,
                                          model=model,
                                          one_class_test=patient_config["one_class_test"],
                                          device=device)

        results["patient_id"].append(patient_id)
        results["ROC AUC"].append(patient_results['ROC AUC'])
        results["PR AUC"].append(patient_results["PR AUC"])
        results["ROC AUC (filter)"].append(patient_results["ROC AUC (filter)"])
        results["PR AUC (filter)"].append(patient_results["PR AUC (filter)"])

    final_df = pd.DataFrame(results)
    final_df.to_csv(f"submission_test_{datetime.today().date()}.csv")
