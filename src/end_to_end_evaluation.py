import os
import sys
import argparse
import json

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from datetime import datetime
from scipy.stats import norm
from sklearn.metrics import precision_recall_curve, roc_curve, auc

from models.convolutional_autoencoder import Autoencoder, UNet
from datasets.dataset import PatientDataset
from callbacks.callbacks import EarlyStopping
import utils.parse as parser


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


def train_loop(train_dset, val_dset, model, epochs, batch_size, patience, learning_rate, pt_file, device, num_workers):

    # Initialize dataloaders & optimizers
    model = model.to(device)
    train_dloader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dloader = DataLoader(val_dset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    optim = Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = CosineAnnealingLR(optimizer=optim, T_max=epochs, eta_min=1e-5)
    ear_stopping = EarlyStopping(patience=patience, verbose=True, path=pt_file)
    loss_fn = nn.MSELoss()
    loss_fn = loss_fn.to(device=device)

    train_loss, val_loss, best_val_loss = 0., 0., np.inf
    _padding = len(str(epochs + 1))

    for epoch in range(1, epochs + 1):
        model.train()
        with tqdm(train_dloader, unit="batch", leave=False, desc="Training set") as tbatch:
            for d in tbatch:
                # Forward
                org_features, mask = d["features"], d["mask"]
                org_features, mask = org_features.to(device), mask.to(device)
                reco_features = model(org_features) * mask

                loss = loss_fn(org_features, reco_features)
                train_loss += loss.item()

                # Backward
                optim.zero_grad()
                loss.backward()
                optim.step()
        train_loss /= len(train_dloader)
        lr_scheduler.step()

        # Validation loop
        model.eval()
        with torch.no_grad():
            with tqdm(val_dloader, unit="batch", leave=False, desc="Validation set") as vbatch:
                for d in vbatch:
                    # Forward
                    org_features, mask = d["features"], d["mask"]
                    org_features, mask = org_features.to(device), mask.to(device)
                    reco_features = model(org_features) * mask

                    loss = loss_fn(org_features, reco_features)
                    val_loss += loss.item()
        val_loss /= len(val_dloader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        print(f"Epoch {epoch:<{_padding}}/{epochs}. Train Loss: {train_loss:.3f}. Val Loss: {val_loss:.3f}")

        ear_stopping(val_loss, model, epoch)
        if ear_stopping.early_stop:
            print("Early Stopping.")
            return best_val_loss
        train_loss, val_loss = 0.0, 0.0

    return best_val_loss


def validation_loop(train_dset, test_dset, model, device):
    test_dloader = DataLoader(test_dset, batch_size=1, shuffle=False, drop_last=False, num_workers=1)
    train_dloader = DataLoader(train_dset, batch_size=1, shuffle=False, drop_last=False, num_workers=1)
    loss_fn = nn.MSELoss().to(device=device)
    model = model.to(device)

    # Loop over train and determine distribution
    train_losses = []
    model.eval()
    with torch.no_grad():
        for d in train_dloader:
            # Inference
            features, mask = d['features'], d['mask']
            features, mask = features.to(device), mask.to(device)
            reco_features = model(features) * mask

            loss = loss_fn(features, reco_features)
            train_losses.append(loss.item())
    # Calculate mean & std and fit Normal distribution
    mu, std = np.mean(train_losses), np.std(train_losses)

    val_losses, splits, days, labels = [], [], [], []
    with torch.no_grad():
        for d in test_dloader:
            # Inference
            features, mask = d['features'], d['mask']
            features, mask = features.to(device), mask.to(device)
            reco_features = model(features) * mask

            loss = loss_fn(features, reco_features)
            val_losses.append(loss.item())
            splits.append(d['split'][0])
            days.append(d['day_index'].item())
            labels.append(d['label'].item())

    df = pd.DataFrame({"split": splits, "day_index": days, "val_loss": val_losses, "label": labels})
    res = df.groupby(by=["split", "day_index"]).mean()
    res['label'] = res['label'].astype(int)
    res['anomaly_scores'] = res['val_loss'].map(lambda x: 1 - norm.pdf(x, mu, std) / norm.pdf(mu, mu, std))
    res['anomaly_scores_random'] = np.random.random(size=res.shape[0])

    unique_splits = np.unique(splits).tolist()

    return {"scores": res, "Distribution Loss (mean)": mu, "Distribution Loss (std)": std, "split": unique_splits}


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
            rec_loss_train = train_loop(train_dset=train_dset,
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
            val_results = validation_loop(train_dset, test_dset, model, device)

            results["Distribution Loss (mean)"].append(val_results['Distribution Loss (mean)'])
            results["Distribution Loss (std)"].append(val_results['Distribution Loss (std)'])

            val_losses = val_results['scores']['val_loss'].to_numpy()
            labels = val_results['scores']['label'].to_numpy()
            anomaly_scores = val_results['scores']['anomaly_scores'].to_numpy()
            anomaly_scores_random = val_results['scores']['anomaly_scores_random'].to_numpy()

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

            results['Mean anomaly score (non relapsed)'].append(np.mean(anomaly_scores[labels == 0]))
            results['Mean anomaly score (relapsed)'].append(np.mean(anomaly_scores[labels == 1]))

            results['Rec Loss (non relapsed)'].append(np.mean(val_losses[labels == 0]))
            results['Rec Loss (relapsed)'].append(np.mean(val_losses[labels == 1]))

            # Write csvs
            final_df = pd.DataFrame(results)
            final_df.to_csv("results_" + str(datetime.today().date()) + ".csv")

            patient_path = parser.get_path(track_id, patient_id)

            for split in val_results['split']:
                filt_df = val_results['scores'].loc[split]
                filt_df.to_csv(os.path.join(patient_path, split, f"results_{model_str}_{datetime.today().date()}.csv"))
