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
from scipy.stats import norm
from sklearn.metrics import precision_recall_curve, roc_curve, auc

from models.convolutional_autoencoder import Autoencoder, UNet
from datasets.dataset import PatientDataset
import utils.parse as parser
from training.loops import autoencoder_train_loop, validation_loop


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='Json training config file.')

    return parser.parse_args()


def get_model(model_str: str):
    if model_str == 'Autoencoder':
        model = Autoencoder()
    elif model_str == 'UNet':
        model = UNet(in_channels=1, out_channels=1)
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
    models = json_config["models"]
    path_of_pt_files = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
                                    json_config['pretrained_models'])

    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Check for One Class Svm test
    if "one_class_test" in json_config.keys() and json_config['one_class_test'] == True:
        results = {
            "Patient_id": [],
            "Model": [],
            "Rec Loss (train)": [],
            "ROC AUC": [],
            "PR AUC": [],
            "ROC AUC (random)": [],
            "PR AUC (random)": [],
            "Total days on train": [],
            "Total days (relapsed)": [],
            "Total days (non relapsed)": []
        }
        one_class_test = True
    else:
        # These are the results to display at the end of the experiment
        results = {
            "Patient_id": [],
            "Model": [],
            "Rec Loss (train)": [],
            "Rec Loss (relapsed)": [],
            "Rec Loss (non relapsed)": [],
            "Distribution Loss (mean)": [],
            "Distribution Loss (std)": [],
            "ROC AUC": [],
            "PR AUC": [],
            "Mean anomaly score (relapsed)": [],
            "Mean anomaly score (non relapsed)": [],
            "ROC AUC (random)": [],
            "PR AUC (random)": [],
            "Total days on train": [],
            "Total days (relapsed)": [],
            "Total days (non relapsed)": []
        }

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
        for model_str in tqdm(models, desc="Validating model", leave=False):
            model = get_model(model_str=model_str)

            # Check for model transfer learning / works when only one model is given
            if "transfer_learning" in json_config.keys():
                model.load_state_dict(torch.load(json_config["transfer_learning"]))
                print("Transfer learning from all data.")

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

            results["Rec Loss (train)"].append(rec_loss_train)

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

            # Get results and write outputs
            val_results = validation_loop(train_dset, test_dset, model, device, one_class_test=one_class_test)

            if not one_class_test:
                results["Distribution Loss (mean)"].append(val_results['Distribution Loss (mean)'])
                results["Distribution Loss (std)"].append(val_results['Distribution Loss (std)'])

                val_losses = val_results['scores']['val_loss'].to_numpy()
                labels = val_results['scores']['label'].to_numpy()
                anomaly_scores = val_results['scores']['anomaly_scores'].to_numpy()
                anomaly_scores_random = val_results['scores']['anomaly_scores_random'].to_numpy()
            else:
                anomaly_scores, labels = val_results['anomaly_scores'], val_results['labels']
                anomaly_scores_random = np.random.random(size=len(anomaly_scores))

            # Compute metrics
            precision, recall, _ = precision_recall_curve(labels, anomaly_scores)

            fpr, tpr, _ = roc_curve(labels, anomaly_scores)

            results["ROC AUC"].append(auc(fpr, tpr))
            results['PR AUC'].append(auc(recall, precision))

            # Compute metrics for random guess
            precision, recall, _ = precision_recall_curve(labels, anomaly_scores_random)
            fpr, tpr, _ = roc_curve(labels, anomaly_scores_random)
            results["ROC AUC (random)"].append(auc(fpr, tpr))
            results['PR AUC (random)'].append(auc(recall, precision))

            results['Total days (non relapsed)'].append(len(labels[labels == 0]))
            results['Total days (relapsed)'].append(len(labels[labels == 1]))

            if not one_class_test:
                results['Mean anomaly score (non relapsed)'].append(np.mean(anomaly_scores[labels == 0]))
                results['Mean anomaly score (relapsed)'].append(np.mean(anomaly_scores[labels == 1]))
                results['Rec Loss (non relapsed)'].append(np.mean(val_losses[labels == 0]))
                results['Rec Loss (relapsed)'].append(np.mean(val_losses[labels == 1]))

            # Write csvs
            final_df = pd.DataFrame(results)
            final_df.to_csv("results_" + str(datetime.today().date()) + ".csv")

            if not one_class_test:
                patient_path = parser.get_path(track_id, patient_id)

                for split in val_results['split']:
                    filt_df = val_results['scores'].loc[split]
                    filt_df.to_csv(
                        os.path.join(patient_path, split, f"results_{model_str}_{datetime.today().date()}.csv"))