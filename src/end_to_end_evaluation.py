import os
import sys
import argparse
import json

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from datetime import datetime
from scipy.signal import medfilt
from sklearn.metrics import precision_recall_curve, roc_curve, auc

from models.convolutional_autoencoder import Autoencoder, Autoencoder_2
from models.anomaly_transformer import *
from models import anomaly_transformer as vits
from datasets.dataset import PatientDataset
import utils.parse as parser
from training.loops import autoencoder_train_loop, validation_loop
from utils.util_funcs import fill_predictions, calculate_roc_pr_auc
from scipy.signal import medfilt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='Json training config file.')

    return parser.parse_args()


def get_model(model_str, window_size, num_layers):
    if model_str == 'Autoencoder':
        model = Autoencoder()
    elif model_str == 'Autoencoder_2':
        num_channels = [1] + [2**(i + 2) for i in range(num_layers)]
        model = Autoencoder_2((window_size, 16), num_channels)
    elif model_str == 'AnomalyTransformer':
        print('vits dict:', vits.__dict__)
        student = vits.__dict__['vit_base'](in_chans=1, img_size=[16, window_size])
        model = FullPipline(student, CLSHead(512, 256), RECHead(768))
    elif model_str == 'AnomalyTransformer_2':
        student = vits.__dict__['vit_base'](in_chans=1, img_size=[16, window_size], depth=num_layers)
        model = FullPipline(student, CLSHead(512, 256), RECHead(768))
    return model


if __name__ == '__main__':

    # Parse args
    print('STarting!')
    args = parse_args()
    with open(args.config, "r") as f:
        json_config = json.load(f)

    feature_mapping = json_config["feature_mapping"]
    track_id = json_config["track_id"]
    patients = json_config["patients"]
    window_size = json_config["window_size"]
    upsampling_size = json_config["upsampling_size"]
    num_layers = json_config["num_layers"]
    model_str = json_config["model"]
    path_of_pt_files = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
                                    json_config['pretrained_models'])
    one_class_test = json_config["one_class_test"]
    test_metrics = json_config["test_metric"]
    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print('json config:', json_config['one_class_test'] == True)

    results = {
        "Patient_id": [],
        "Model": [],
        "Inference Method": [],
        "Rec Loss (train)": [],
        "Distribution Loss (mean)": [],
        "Distribution Loss (std)": [],
        "ROC AUC": [],
        "PR AUC": [],
        "ROC AUC interpolation": [],
        "PR AUC interpolation": [],
        "ROC AUC (random)": [],
        "PR AUC (random)": [],
        "Total days on train": [],
        "Total days (relapsed)": [],
        "Total days (non relapsed)": [],
        "Test metric": []
    }
    if "postprocessing_filters" in json_config.keys():
        for filter_size in json_config["postprocessing_filters"]:
            results[f'median filter ROC AUC ({filter_size})'] = []
            results[f'median filter PR AUC ({filter_size})'] = []
            results[f'mean filter ROC AUC ({filter_size})'] = []
            results[f'mean filter PR AUC ({filter_size})'] = []

    cnt = 0
    for patient_id in tqdm(patients, desc='Evaluating on each patient', total=len(patients)):

        # Initialize patient's dataset and split to train/val -> Same split for each model
        X = parser.get_features(track_id=track_id,
                                patient_id=patient_id,
                                mode="train",
                                extension=json_config['file_format'])

        X_train, X_val = train_test_split(X, test_size=1 - json_config['split_ratio'], random_state=42)

        train_dset = PatientDataset(track_id=track_id,
                                    patient_id=patient_id,
                                    mode="train",
                                    window_size=window_size,
                                    extension=json_config['file_format'],
                                    feature_mapping=feature_mapping,
                                    patient_features=X_train,
                                    from_path=False)

        # Calculate statistics (mean, std) on train and pass to other datasets
        train_dset._cal_statistics()
        mu, std = train_dset.mean, train_dset.std

        val_dset = PatientDataset(track_id=track_id,
                                  patient_id=patient_id,
                                  mode="train",
                                  window_size=window_size,
                                  extension=json_config['file_format'],
                                  feature_mapping=feature_mapping,
                                  patient_features=X_val,
                                  from_path=False)

        val_dset.mean, val_dset.std = mu, std

        test_dset = PatientDataset(track_id=track_id,
                                   patient_id=patient_id,
                                   mode="val",
                                   window_size=window_size,
                                   extension=json_config["file_format"],
                                   feature_mapping=feature_mapping,
                                   from_path=True)

        test_dset.mean, test_dset.std = mu, std

        # Upsample train/val sets
        train_dset._upsample_data(upsample_size=upsampling_size)
        val_dset._upsample_data(upsample_size=upsampling_size)

        model = get_model(model_str, window_size, num_layers[cnt])

        # Check for model transfer learning / works when only one model is given
        if "transfer_learning" in json_config.keys():
            model.load_state_dict(torch.load(json_config["transfer_learning"]))
            print("Transfer learning from all data.")

        if "saved_checkpoint" in json_config.keys() and json_config["saved_checkpoint"][cnt]:
            pt_file = json_config["saved_checkpoint"][cnt]
            rec_loss_train = " "
        else:
            # Get patient's path to store pt files
            pt_file = os.path.join(
                path_of_pt_files,
                f"Track_{track_id}_P{patient_id}_" + model_str + "_" + str(datetime.today().date()) + ".pt")

            # Start training
            rec_loss_train = autoencoder_train_loop(train_dset=train_dset,
                                                    val_dset=val_dset,
                                                    model=model,
                                                    epochs=json_config["epochs"],
                                                    batch_size=json_config["batch_size"],
                                                    patience=json_config["patience"],
                                                    learning_rate=json_config["learning_rate"],
                                                    pt_file=pt_file,
                                                    device=device,
                                                    num_workers=json_config['num_workers'])

        # Load best model and validate
        model.load_state_dict(torch.load(pt_file, map_location=device))
        # Re-initialize training set to fit distribution on losses
        train_dset = PatientDataset(track_id=track_id,
                                    patient_id=patient_id,
                                    mode="train",
                                    window_size=window_size,
                                    extension=json_config["file_format"],
                                    feature_mapping=feature_mapping,
                                    from_path=True)
        train_dset.mean, train_dset.std = mu, std

        results['Total days on train'].append(len(train_dset))

        # Upsample on predictions
        train_dset._upsample_data(upsample_size=json_config["prediction_upsampling"])
        test_dset._upsample_data(upsample_size=json_config["prediction_upsampling"])

        val_results = validation_loop(train_dset,
                                      test_dset,
                                      model,
                                      device,
                                      test_metrics,
                                      patient_id,
                                      one_class_test=one_class_test,
                                      path_of_pt_files=path_of_pt_files)

        for i, result in enumerate(val_results):
            # Update results
            if "postprocessing_filters" in json_config.keys():
                filter_scores = {}
                for filter_size in json_config["postprocessing_filters"]:
                    filter_scores[f'median filter scores ({filter_size})'] = []
                    filter_scores[f'mean filter scores ({filter_size})'] = []
            results["Rec Loss (train)"].append(rec_loss_train)
            results['Model'].append(model_str)
            results['Patient_id'].append(patient_id)
            if not ('Distribution Loss (mean)' in result):
                results['Test metric'].append(result["test_metric"])
                results["Inference Method"].append("OC-SVM")
                results["Distribution Loss (mean)"].append(" ")
                results["Distribution Loss (std)"].append(" ")
                anomaly_scores, labels = result['anomaly_scores'], result['labels']
                # Write csvs with predictions
                patient_path = parser.get_path(track_id, patient_id)
                df = pd.DataFrame({
                    "anomaly_scores": anomaly_scores,
                    "label": labels,
                    "split_day": result["split_days"]
                })

                df['split'] = [x.split("_")[0] + "_" + str(x.split("_")[1]) for x in df['split_day']]
                df['day_index'] = [int(x.split("_")[3]) for x in df["split_day"]]

                anomaly_scores_inter, labels_inter = [], []
                anomaly_scores, labels = result['anomaly_scores'], result['labels']
                for sp in df['split'].unique():
                    mode, num = sp.split("_")[0], int(sp.split("_")[1])
                    filt_df = df[df['split'] == sp]

                    filt_df = fill_predictions(track_id=track_id,
                                               patient_id=patient_id,
                                               anomaly_scores=filt_df['anomaly_scores'],
                                               split=sp,
                                               days=filt_df['day_index'])
                    anomaly_scores_temp = filt_df['anomaly_scores'].to_numpy()
                    if "postprocessing_filters" in json_config.keys():
                        for filter_size in json_config["postprocessing_filters"]:
                            # Median filter
                            median_anomaly_scores = medfilt(anomaly_scores_temp, filter_size)
                            filter_scores[f'median filter scores ({filter_size})'].append(median_anomaly_scores)

                            # Mean filter
                            mean_anomaly_scores = np.convolve(anomaly_scores_temp,
                                                              np.ones(filter_size) / filter_size, "same")

                            filter_scores[f'mean filter scores ({filter_size})'].append(mean_anomaly_scores)

                    path_to_save = parser.get_path(track=track_id, patient=patient_id, mode=mode, num=num)
                    filt_df.to_csv(os.path.join(path_to_save, f"results_{model_str}_{datetime.today().date()}.csv"))

                    anomaly_scores_inter.append(anomaly_scores_temp)
                    labels_inter.append(filt_df['relapse'].to_numpy())
            else:
                results['Test metric'].append(result["test_metric"])
                results["Distribution Loss (mean)"].append(result['Distribution Loss (mean)'])
                results["Distribution Loss (std)"].append(result['Distribution Loss (std)'])
                results["Inference Method"].append("MSE Loss")

                val_losses = result['scores']['val_loss'].to_numpy()

                labels = result['scores']['label'].to_numpy()
                anomaly_scores = result['scores']['anomaly_scores'].to_numpy()

                anomaly_scores_inter, labels_inter = [], []

                # Write csvs with predictions
                patient_path = parser.get_path(track_id, patient_id)
                for split in result['split']:
                    filt_df = result['scores'].loc[split].reset_index(names="day_index")
                    filt_df = fill_predictions(track_id=track_id,
                                               patient_id=patient_id,
                                               anomaly_scores=filt_df['anomaly_scores'],
                                               split=split,
                                               days=filt_df["day_index"])

                    anomaly_scores_temp = filt_df['anomaly_scores']
                    if "postprocessing_filters" in json_config.keys():
                        for filter_size in json_config["postprocessing_filters"]:
                            # Median filter
                            median_anomaly_scores = medfilt(anomaly_scores_temp, filter_size)
                            filter_scores[f'median filter scores ({filter_size})'].append(median_anomaly_scores)

                            # Mean filter
                            mean_anomaly_scores = np.convolve(anomaly_scores_temp,
                                                              np.ones(filter_size) / filter_size, "same")

                            filter_scores[f'mean filter scores ({filter_size})'].append(mean_anomaly_scores)
                    filt_df.to_csv(
                        os.path.join(patient_path, split, f"results_{model_str}_{datetime.today().date()}.csv"))

                    anomaly_scores_inter.append(anomaly_scores_temp)
                    labels_inter.append(filt_df['relapse'])

            anomaly_scores_inter, labels_inter = np.concatenate(anomaly_scores_inter), np.concatenate(labels_inter)

            anomaly_scores_random = np.random.random(size=len(anomaly_scores))

            # Compute metrics
            # without interpolation
            precision, recall, _ = precision_recall_curve(labels, anomaly_scores)
            fpr, tpr, _ = roc_curve(labels, anomaly_scores)
            results["ROC AUC"].append(auc(fpr, tpr))
            results['PR AUC'].append(auc(recall, precision))

            #with interpolation
            if "postprocessing_filters" in json_config.keys():
                for filter_size in json_config["postprocessing_filters"]:
                    # Median
                    median_anomaly_scores = np.concatenate(filter_scores[f'median filter scores ({filter_size})'])
                    scores = calculate_roc_pr_auc(median_anomaly_scores, labels_inter)
                    results[f'median filter ROC AUC ({filter_size})'].append(scores["ROC AUC"])
                    results[f'median filter PR AUC ({filter_size})'].append(scores["PR AUC"])
                    # Mean filter
                    mean_anomaly_scores = np.concatenate(filter_scores[f'mean filter scores ({filter_size})'])
                    scores = calculate_roc_pr_auc(mean_anomaly_scores, labels_inter)
                    results[f'mean filter ROC AUC ({filter_size})'].append(scores["ROC AUC"])
                    results[f'mean filter PR AUC ({filter_size})'].append(scores["PR AUC"])
            precision, recall, _ = precision_recall_curve(labels_inter, anomaly_scores_inter)
            fpr, tpr, _ = roc_curve(labels_inter, anomaly_scores_inter)
            results["ROC AUC interpolation"].append(auc(fpr, tpr))
            results['PR AUC interpolation'].append(auc(recall, precision))

            # Compute metrics for random guess
            scores = calculate_roc_pr_auc(anomaly_scores_random, labels)
            results["ROC AUC (random)"].append(scores["ROC AUC"])
            results['PR AUC (random)'].append(scores["PR AUC"])

            results['Total days (non relapsed)'].append(len(labels[labels == 0]))
            results['Total days (relapsed)'].append(len(labels[labels == 1]))

        # Write csvs
        final_df = pd.DataFrame(results)
        final_df.to_csv("results_" + str(datetime.today().date()) + ".csv")

        cnt += 1
