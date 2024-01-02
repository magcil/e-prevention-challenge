import os
import sys
import argparse
import json
import functools

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
import optuna
from scipy.signal import medfilt

from models.convolutional_autoencoder import Autoencoder, UNet, Autoencoder_2
from models.anomaly_transformer import *
from models import anomaly_transformer as vits
from datasets.dataset import PatientDataset
from callbacks.callbacks import EarlyStoppingAUC
import utils.parse as parser
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
    return model


def train_loop(train_dset, whole_train, val_dset, test_dset, model, epochs, batch_size, patience, learning_rate,
               scheduler_name, pt_file, device, num_workers):

    # Initialize dataloaders & optimizers
    model = model.to(device)
    train_dloader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dloader = DataLoader(val_dset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
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
                reco_features, _ = model(org_features)
                reco_features = reco_features * mask

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
                    reco_features, _ = model(org_features)
                    reco_features = reco_features * mask

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
        df = validation_loop(whole_train, test_dset, model, device, batch_size=batch_size, num_workers=num_workers)

        # Fill predictions and apply filters
        df['split'] = [x.split("_")[0] + "_" + str(x.split("_")[1]) for x in df['split_days']]
        df['day_index'] = [int(x.split("_")[3]) for x in df["split_days"]]

        anomaly_scores, labels = [], []

        for sp in df['split'].unique():
            filt_df = df[df['split'] == sp]

            filt_df = fill_predictions(track_id=track_id,
                                       patient_id=patient_id,
                                       anomaly_scores=filt_df['anomaly_scores'],
                                       split=sp,
                                       days=filt_df['day_index'])

            anomaly_scores.append(filt_df['anomaly_scores'].to_numpy())
            labels.append(filt_df['relapse'].to_numpy())

        labels, anomaly_scores = np.concatenate(labels), np.concatenate(anomaly_scores)

        # Compute metrics
        scores = calculate_roc_pr_auc(anomaly_scores, labels)

        score = (scores["ROC AUC"] + scores["PR AUC"]) / 2

        ear_stopping(score, model, epoch)
        if ear_stopping.early_stop:
            print("Early Stopping.")
            return ear_stopping.best_score, ear_stopping.best_model, ear_stopping.best_epoch
        train_loss, val_loss = 0.0, 0.0

    return ear_stopping.best_score, ear_stopping.best_model, ear_stopping.best_epoch


def validation_loop(train_dset, test_dset, model, device, batch_size, num_workers):
    test_dloader = DataLoader(test_dset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    train_dloader = DataLoader(train_dset,
                               batch_size=batch_size,
                               shuffle=False,
                               drop_last=False,
                               num_workers=num_workers)
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
            reco_features, emb = model(features)
            reco_features = reco_features * mask

            loss = (reco_features - features) ** 2
            loss = np.mean(loss.cpu().numpy(), axis=(1, 2, 3))
            train_losses.append(loss)

    # Calculate mean & std and fit Normal distribution
    train_losses = np.concatenate(train_losses)
    mu, std = np.mean(train_losses), np.std(train_losses)

    val_losses, splits, days, labels = [], [], [], []
    with torch.no_grad():
        for d in test_dloader:
            # Inference
            features, mask = d['features'], d['mask']
            features, mask = features.to(device), mask.to(device)
            reco_features, emb = model(features)
            reco_features = reco_features * mask

            features, reco_features = features.cpu().numpy(), reco_features.cpu().numpy()

            loss = np.mean((features - reco_features)**2, axis=(1, 2, 3))

            val_losses.append(loss)
            splits.append(d['split'])
            days.append(d['day_index'].numpy())
            labels.append(d['label'].numpy())

        val_losses, splits, days, labels = np.concatenate(val_losses), np.concatenate(splits), np.concatenate(
            days), np.concatenate(labels)

    df = pd.DataFrame({"split": splits, "day_index": days, "val_loss": val_losses, "label": labels})
    res = df.groupby(by=["split", "day_index"]).mean()
    res['label'] = res['label'].astype(int)
    res['anomaly_scores'] = res['val_loss'].map(lambda x: 1 - norm.pdf(x, mu, std) / norm.pdf(mu, mu, std))

    res.reset_index(names=["split", "day_index"], inplace=True)
    res['split_days'] = [split + "_day_" + str(day) for split, day in zip(res['split'], res['day_index'])]

    return res


def objective(trial, track_id, patient_id, json_config, window_size, train_dset, whole_train_dset, val_dset, test_dset):
    #window_size = trial.suggest_categorical('window_size', [48, 128, 160])
    #upsampling_size = trial.suggest_categorical('upsampling_size', [50, 100, 200, 500])

    num_layers = trial.suggest_int('num_layers', 5, 15)
    # Ensure that num_units correspond to the specified num_layers
    #num_channels = [1] + [2 ** (i + 2) for i in range(num_layers)]
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1e-3)
    #scheduler_name = trial.suggest_categorical('scheduler', [None, 'StepLR', 'CosineAnnealingLR', 'ReduceLROnPlateau'])
    scheduler_name = 'CosineAnnealingLR'

    path_of_pt_files = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
                                    json_config['pretrained_models'])
    # Get patient's path to store pt files
    pt_file = os.path.join(path_of_pt_files,
                           f"Track_{track_id}_P{patient_id}_" + "Transformer" + "_" + str(datetime.today()) + ".pt")

    student = vits.__dict__['vit_base'](in_chans=1, img_size=[16, window_size], depth=num_layers)
    model = FullPipline(student, CLSHead(512, 256), RECHead(768))
    # Start training
    best_score, model, best_epoch = train_loop(train_dset=train_dset,
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
                                               device=device,
                                               num_workers=json_config['num_workers'])
    print("best score (average aucs): ", best_score)

    # Get results and write outputs
    df = validation_loop(whole_train_dset,
                         test_dset,
                         model,
                         device,
                         batch_size=json_config["batch_size"],
                         num_workers=json_config['num_workers'])

    if "postprocessing_filters" in json_config.keys():
        filter_scores = {}
        for filter_size in json_config["postprocessing_filters"]:
            filter_scores[f'median filter scores ({filter_size})'] = []
            filter_scores[f'mean filter scores ({filter_size})'] = []

    df['split'] = [x.split("_")[0] + "_" + str(x.split("_")[1]) for x in df['split_days']]
    df['day_index'] = [int(x.split("_")[3]) for x in df["split_days"]]

    anomaly_scores, labels = [], []

    for sp in df['split'].unique():
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
                mean_anomaly_scores = np.convolve(anomaly_scores_temp, np.ones(filter_size) / filter_size, "same")

                filter_scores[f'mean filter scores ({filter_size})'].append(mean_anomaly_scores)

        anomaly_scores.append(anomaly_scores_temp)
        labels.append(filt_df['relapse'].to_numpy())

    anomaly_scores, labels = np.concatenate(anomaly_scores), np.concatenate(labels)
    anomaly_scores_random = np.repeat(0.5, len(anomaly_scores))

    scores = calculate_roc_pr_auc(anomaly_scores, labels)

    score = (scores["ROC AUC"] + scores["PR AUC"]) / 2

    scores_random = calculate_roc_pr_auc(anomaly_scores_random, labels)

    # Additional information you want to return along with the validation loss
    additional_info = {
        'ROC AUC': scores["ROC AUC"],
        'PR AUC': scores["PR AUC"],
        "ROC AUC (random)": scores_random["ROC AUC"],
        "PR AUC (random)": scores_random["PR AUC"],
        'Best epoch': best_epoch,
        "model": model
    }

    if "postprocessing_filters" in json_config.keys():
        for filter_size in json_config["postprocessing_filters"]:
            # Median
            median_anomaly_scores = np.concatenate(filter_scores[f'median filter scores ({filter_size})'])
            scores = calculate_roc_pr_auc(median_anomaly_scores, labels)

            additional_info[f'median filter ROC AUC ({filter_size})'] = scores["ROC AUC"]
            additional_info[f'median filter PR AUC ({filter_size})'] = scores["PR AUC"]

            # Mean filter
            mean_anomaly_scores = np.concatenate(filter_scores[f'mean filter scores ({filter_size})'])
            scores = calculate_roc_pr_auc(mean_anomaly_scores, labels)
            additional_info[f'mean filter ROC AUC ({filter_size})'] = scores["ROC AUC"]
            additional_info[f'mean filter PR AUC ({filter_size})'] = scores["PR AUC"]

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

        print(f"\n\n{10*'*'} Patient {patient_id} {10*'*'}\n\n")

        ################### Load data ###################################
        window_size = 32
        upsampling_size = 120
        # Initialize patient's dataset and split to train/val -> Same split for each model
        X = parser.get_features(track_id=track_id,
                                patient_id=patient_id,
                                mode="train",
                                extension=json_config["file_format"])

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

        #################### Run tuner #################################
        objective_with_args = functools.partial(objective,
                                                track_id=track_id,
                                                patient_id=patient_id,
                                                json_config=json_config,
                                                window_size=window_size,
                                                train_dset=train_dset,
                                                whole_train_dset=whole_train_dset,
                                                val_dset=val_dset,
                                                test_dset=test_dset)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective_with_args, n_trials=5)

        #save best model
        path_of_pt_files = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
                                        json_config['pretrained_models'])
        model_name = os.path.join(path_of_pt_files, 'p' + str(patient_id) + '_transformer_best_model.pth')
        best_model = study.best_trial.user_attrs['additional_info']["model"]
        torch.save(best_model.state_dict(), model_name)

        best_params = study.best_params
        print("Best Hyperparameters:", best_params)
        best_score = study.best_value
        best_additional_info = study.best_trial.user_attrs['additional_info']
        print("Best Score:", best_score)
        print("Best Additional Info:", best_additional_info)
