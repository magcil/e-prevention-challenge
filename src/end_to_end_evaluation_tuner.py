import os
import sys
import argparse
import json

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from datetime import datetime
from scipy.stats import norm
from sklearn.metrics import precision_recall_curve, roc_curve, auc

from models.convolutional_autoencoder import Autoencoder, UNet, Autoencoder_2
from datasets.dataset import PatientDataset
from callbacks.callbacks import EarlyStoppingAUC
import utils.parse as parser
import optuna
import functools

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



def train_loop(train_dset, whole_train, val_dset, test_dset, model,
               epochs, batch_size, patience, learning_rate, scheduler_name, pt_file, device):

    # Initialize dataloaders & optimizers
    model = model.to(device)
    train_dloader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dloader = DataLoader(val_dset, batch_size=batch_size, shuffle=False, num_workers=4)
    optim = Adam(model.parameters(), lr=learning_rate)
    if scheduler_name == 'StepLR':
        scheduler = StepLR(optim, step_size=30, gamma=0.5)
    elif scheduler_name == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optim, T_max=epochs, eta_min=1e-5)
    elif scheduler_name == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optim, 'min', patience=5)

    ear_stopping = EarlyStoppingAUC(patience=patience, verbose=True, path=pt_file)
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

        # Scheduler step
        if scheduler_name is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        print(f"Epoch {epoch:<{_padding}}/{epochs}. Train Loss: {train_loss:.3f}. Val Loss: {val_loss:.3f}")
        # Get results and write outputs
        val_results = validation_loop(whole_train, test_dset, model, device)

        labels = val_results['scores']['label'].tolist()
        anomaly_scores = val_results['scores']['anomaly_scores'].to_numpy()

        # Compute metrics
        precision, recall, _ = precision_recall_curve(labels, anomaly_scores)

        fpr, tpr, _ = roc_curve(labels, anomaly_scores)
        score = (auc(fpr, tpr) + auc(recall, precision)) / 2
        ear_stopping(score, model, epoch)
        if ear_stopping.early_stop:
            print("Early Stopping.")
            return ear_stopping.best_score, ear_stopping.best_model
        train_loss, val_loss = 0.0, 0.0

    return ear_stopping.best_score, ear_stopping.best_model


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


def objective(trial, track_id, patient_id, json_config, feature_mapping):
    #window_size = trial.suggest_categorical('window_size', [48, 128, 160])
    #upsampling_size = trial.suggest_categorical('upsampling_size', [50, 100, 200, 500])
    window_size = 32
    upsampling_size = 120
    #latent_dim = trial.suggest_categorical('latent_dim', [32, 64, 128, 256])
    num_layers = trial.suggest_int('num_layers', 3, 6)
    # Ensure that num_units correspond to the specified num_layers
    num_channels = [1] + [2 ** (i + 2) for i in range(num_layers)]
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1e-3)
    #scheduler_name = trial.suggest_categorical('scheduler', [None, 'StepLR', 'CosineAnnealingLR', 'ReduceLROnPlateau'])
    scheduler_name = 'CosineAnnealingLR'
    # Initialize patient's dataset and split to train/val -> Same split for each model
    X = parser.get_features(track_id=track_id, patient_id=patient_id, mode="train")

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

    # Re-initialize training set to fit distribution on losses
    whole_train_dset = PatientDataset(track_id=track_id,
                                      patient_id=patient_id,
                                      mode="train",
                                      window_size=window_size,
                                      extension=json_config["file_format"],
                                      feature_mapping=feature_mapping,
                                      from_path=True)
    whole_train_dset.mean, whole_train_dset.std = mu, std
    # Upsample train/val sets
    train_dset._upsample_data(upsample_size=upsampling_size)
    val_dset._upsample_data(upsample_size=upsampling_size)
    test_dset._upsample_data(upsample_size=upsampling_size)
    whole_train_dset._upsample_data(upsample_size=upsampling_size)

    path_of_pt_files = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
                                    json_config['pretrained_models'])
    # Get patient's path to store pt files
    pt_file = os.path.join(path_of_pt_files,
                           f"Track_{track_id}_P{patient_id}_" + "Autoencoder_2" + "_" + str(datetime.today()) + ".pt")
    model = Autoencoder_2((window_size, 16), num_channels)
    # Start training
    best_score, model = train_loop(train_dset=train_dset,
                                   whole_train=whole_train_dset,
                                   val_dset=val_dset,
                                   test_dset=test_dset,
                                   model=model,
                                   epochs=json_config["epochs"],
                                   batch_size=json_config["batch_size"],
                                   patience=json_config["patience"],
                                   learning_rate=learning_rate,
                                   scheduler_name=scheduler_name,
                                   pt_file=pt_file,
                                   device=device)
    print("best score: ", best_score)
    # Save the best model within the objective function
    torch.save(model, 'best_model.pth')

    # Get results and write outputs
    val_results = validation_loop(whole_train_dset, test_dset, model, device)

    labels = val_results['scores']['label'].tolist()
    anomaly_scores = val_results['scores']['anomaly_scores'].to_numpy()
    anomaly_scores_random = val_results['scores']['anomaly_scores_random'].to_numpy()

    # Compute metrics
    precision, recall, _ = precision_recall_curve(labels, anomaly_scores)

    fpr, tpr, _ = roc_curve(labels, anomaly_scores)
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall, precision)

    score = (roc_auc + pr_auc) / 2

    precision, recall, _ = precision_recall_curve(labels, anomaly_scores_random)
    fpr, tpr, _ = roc_curve(labels, anomaly_scores_random)
    random_roc_auc = auc(fpr, tpr)
    random_pr_auc = auc(recall, precision)


    # Additional information you want to return along with the validation loss
    additional_info = {'ROC AUC': roc_auc, 'PR AUC': pr_auc, "ROC AUC (random)": random_roc_auc, "PR AUC (random)": random_pr_auc}
    trial.set_user_attr('additional_info', additional_info)  # Store additional_info in user_attrs
    trial.report(score, step=trial.number)
    if trial.should_prune():
        raise optuna.TrialPruned()
    return score



if __name__ == '__main__':

    # Parse args
    args = parse_args()
    with open(args.config, "r") as f:
        json_config = json.load(f)

    feature_mapping = json_config["feature_mapping"]
    track_id = json_config["track_id"]
    patients = json_config["patients"]

    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"


    for patient_id in tqdm(patients, desc='Evaluating on each patient', total=len(patients)):
        objective_with_args = functools.partial(objective, track_id=track_id, patient_id=patient_id,
                                                json_config=json_config, feature_mapping=feature_mapping)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective_with_args, n_trials=5)

        best_params = study.best_params
        print("Best Hyperparameters:", best_params)
        best_score = study.best_value
        best_additional_info = study.best_trial.user_attrs['additional_info']
        print("Best Score:", best_score)
        print("Best Additional Info:", best_additional_info)