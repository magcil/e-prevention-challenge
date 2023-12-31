import os
import sys
import argparse
import json

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from datetime import datetime
from scipy.signal import medfilt

from models.convolutional_autoencoder import Autoencoder, UNet
from models.anomaly_transformer import *
from models import anomaly_transformer as vits
from datasets.dataset import PatientDataset
import utils.parse as parser
from training.loops import autoencoder_train_loop, validation_loop
from utils.util_funcs import fill_predictions, calculate_roc_pr_auc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='Json training config file.')

    return parser.parse_args()


def get_model(model_str: str):
    if model_str == 'Autoencoder':
        model = Autoencoder()
    elif model_str == 'UNet':
        model = UNet(in_channels=1, out_channels=1)
    elif model_str == 'AnomalyTransformer':
        print('vits dict:', vits.__dict__)
        student = vits.__dict__['vit_base'](in_chans=1, img_size=[16, 32])
        model = FullPipline(student, CLSHead(512, 256), RECHead(768))
    return model


if __name__ == '__main__':

    # Parse args
    args = parse_args()
    with open(args.config, "r") as f:
        json_config = json.load(f)

    feature_mapping = json_config["feature_mapping"]
    track_id = json_config["track_id"]
    patients = json_config["patients"]
    window_size = json_config["window_size"]
    upsampling_size = json_config["upsampling_size"]
    path_of_pt_files = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
                                    json_config['pretrained_models'])

    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = {
        "Patient_id": [],
        "Model": [],
        "one_class_test": [],
        "ROC AUC": [],
        "PR AUC": [],
        "ROC AUC (random)": [],
        "PR AUC (random)": [],
        "Total days on train": [],
        "Total days (relapsed)": [],
        "Total days (non relapsed)": []
    }

    if "postprocessing_filters" in json_config.keys():
        for filter_size in json_config['postprocessing_filters']:
            results[f"median filter ROC AUC ({filter_size})"] = []
            results[f"median filter PR AUC ({filter_size})"] = []
            results[f"mean filter ROC AUC ({filter_size})"] = []
            results[f"mean filter PR AUC ({filter_size})"] = []

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

        # Train and validate for each model
        model_str = json_config["patients_config"][f"P{patient_id}"]["model"]
        model = get_model(model_str=json_config["patients_config"][f"P{patient_id}"]["model"])

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

        # Update results
        results['Total days on train'].append(len(train_dset))
        results['Model'].append(model_str)
        results['Patient_id'].append(patient_id)

        # Upsample on predictions
        train_dset._upsample_data(upsample_size=json_config["prediction_upsampling"])
        test_dset._upsample_data(upsample_size=json_config["prediction_upsampling"])

        val_results = validation_loop(
            train_dset=train_dset,
            test_dset=test_dset,
            model=model,
            one_class_test=json_config['patients_config'][f"P{patient_id}"]['one_class_test'],
            device=device)

        results['one_class_test'].append(json_config['patients_config'][f"P{patient_id}"]['one_class_test'])

        anomaly_scores, labels = val_results['anomaly_scores'], val_results['labels']

        if "postprocessing_filters" in json_config.keys():
            filter_scores = {}
            for filter_size in json_config["postprocessing_filters"]:
                filter_scores[f'median filter scores ({filter_size})'] = []
                filter_scores[f'mean filter scores ({filter_size})'] = []

        # Write csvs with predictions
        patient_path = parser.get_path(track_id, patient_id)
        df = pd.DataFrame({
            "anomaly_scores": anomaly_scores,
            "label": labels,
            "split_day": val_results["split_days"]
        })
        df['split'] = [x.split("_")[0] + "_" + str(x.split("_")[1]) for x in df['split_day']]
        df['day_index'] = [int(x.split("_")[3]) for x in df["split_day"]]

        anomaly_scores, labels = [], []

        for sp in df['split'].unique():
            mode, num = sp.split("_")[0], int(sp.split("_")[1])
            filt_df = df[df['split'] == sp]

            filt_df = fill_predictions(track_id=track_id,
                                        patient_id=patient_id,
                                        anomaly_scores=filt_df['anomaly_scores'],
                                        split=sp,
                                        days=filt_df['day_index'])

            path_to_save = parser.get_path(track=track_id, patient=patient_id, mode=mode, num=num)
            filt_df.to_csv(os.path.join(path_to_save, f"results_{model_str}_{datetime.today().date()}.csv"))

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

                    filt_df[f'median filter scores ({filter_size})'] = median_anomaly_scores
                    filt_df[f'mean filter scores ({filter_size})'] = mean_anomaly_scores


            filt_df.to_csv(os.path.join(path_to_save, f"results_{model_str}_{datetime.today().date()}.csv"))

            anomaly_scores.append(anomaly_scores_temp)
            labels.append(filt_df['relapse'].to_numpy())

        anomaly_scores, labels = np.concatenate(anomaly_scores), np.concatenate(labels)
        anomaly_scores_random = np.repeat(0.5, len(anomaly_scores))

        scores = calculate_roc_pr_auc(anomaly_scores, labels)

        results["ROC AUC"].append(scores["ROC AUC"])
        results['PR AUC'].append(scores["PR AUC"])

        # Compute metrics for random guess
        scores = calculate_roc_pr_auc(anomaly_scores_random, labels)
        results["ROC AUC (random)"].append(scores["ROC AUC"])
        results['PR AUC (random)'].append(scores["PR AUC"])

        results['Total days (non relapsed)'].append(len(labels[labels == 0]))
        results['Total days (relapsed)'].append(len(labels[labels == 1]))

        # Use Median/Mean smoothing
        if "postprocessing_filters" in json_config.keys():
            for filter_size in json_config["postprocessing_filters"]:
                # Use median filtering
                median_anomaly_scores = np.concatenate(filter_scores[f'median filter scores ({filter_size})'])
                # Get scores and append
                scores = calculate_roc_pr_auc(median_anomaly_scores, labels)
                results[f"median filter ROC AUC ({filter_size})"].append(scores["ROC AUC"])
                results[f"median filter PR AUC ({filter_size})"].append(scores["PR AUC"])

                # Use mean filtering
                mean_anomaly_scores = np.concatenate(filter_scores[f'mean filter scores ({filter_size})'])
                scores = calculate_roc_pr_auc(mean_anomaly_scores, labels)
                results[f"mean filter ROC AUC ({filter_size})"].append(scores["ROC AUC"])
                results[f"mean filter PR AUC ({filter_size})"].append(scores["PR AUC"])

        # Write csvs
        final_df = pd.DataFrame(results)
        final_df.to_csv("results_" + str(datetime.today().date()) + "upsampling_120_bs128_ws32_depth12_fewer.csv")
