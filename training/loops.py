import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from sklearn.metrics import accuracy_score, f1_score
import torch.nn.functional as TF

from callbacks.callbacks import EarlyStopping
from utils.util_funcs import svm_score


def autoencoder_train_loop(train_dset, val_dset, model, epochs, batch_size, patience, learning_rate, pt_file, device,
                           num_workers):

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
                reco_features, _ = model(org_features)
                reco_features = reco_features * mask

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
                    reco_features, _ = model(org_features)
                    reco_features = reco_features * mask

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


def validation_loop(train_dset, test_dset, model, device, batch_size, num_workers, one_class_test=False):
    test_dloader = DataLoader(test_dset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    train_dloader = DataLoader(train_dset,
                               batch_size=batch_size,
                               shuffle=False,
                               drop_last=False,
                               num_workers=num_workers)

    model = model.to(device)

    # Check if one class SVM test is true
    if one_class_test:
        detector = OneClassSVM()
        scaler = StandardScaler()

    # Loop over train and determine distribution
    train_losses, train_embeddings = [], []
    model.eval()
    with torch.no_grad():
        for d in train_dloader:
            # Inference
            features, mask = d['features'], d['mask']
            features, mask = features.to(device), mask.to(device)
            reco_features, emb = model(features)
            reco_features = reco_features * mask

            if one_class_test:
                emb = torch.flatten(emb, start_dim=1)
                train_embeddings.append(emb.cpu().numpy())

            loss = (reco_features - features)**2
            loss = np.mean(loss.cpu().numpy(), axis=(1, 2, 3))
            train_losses.append(loss)
    # Calculate mean & std and fit Normal distribution
    train_losses = np.concatenate(train_losses)
    mu, std = np.mean(train_losses), np.std(train_losses)

    # Fit One class SVM
    if one_class_test:
        train_embeddings = np.concatenate(train_embeddings)
        scaler.fit(train_embeddings)
        train_embeddings = scaler.transform(train_embeddings)
        detector.fit(train_embeddings)

        test_embeddings = []

    val_losses, splits, days, labels = [], [], [], []
    with torch.no_grad():
        for d in test_dloader:
            # Inference
            features, mask = d['features'], d['mask']
            features, mask = features.to(device), mask.to(device)
            reco_features, emb = model(features)
            reco_features = reco_features * mask

            if one_class_test:
                emb = torch.flatten(emb, start_dim=1)
                test_embeddings.append(emb.cpu().numpy())

            loss = (features - reco_features)**2
            loss = np.mean(loss.cpu().numpy(), axis=(1, 2, 3))
            val_losses.append(loss)
            splits.append(d['split'])
            days.append(d['day_index'].numpy())
            labels.append(d['label'].numpy())

    splits, days, labels, val_losses = np.concatenate(splits), np.concatenate(days), np.concatenate(
        labels), np.concatenate(val_losses)
    # Exract anomaly score with MSE
    if not one_class_test:
        df = pd.DataFrame({"split": splits, "day_index": days, "val_loss": val_losses, "labels": labels})
        res = df.groupby(by=["split", "day_index"]).mean()
        res['labels'] = res['labels'].astype(int)
        res['anomaly_scores'] = res['val_loss'].map(lambda x: 1 - norm.pdf(x, mu, std) / norm.pdf(mu, mu, std))

        res.reset_index(names=["split", "day_index"], inplace=True)
        res['split_days'] = [split + "_day_" + str(day) for split, day in zip(res['split'], res['day_index'])]

        return {
            "anomaly_scores": res["anomaly_scores"].to_numpy(),
            "labels": res['labels'].to_numpy(),
            "split_days": res['split_days'].to_list()
        }

    # Else with embeddings
    else:
        test_embeddings = scaler.transform(np.concatenate(test_embeddings))

        preds = detector.predict(test_embeddings)
        dist_from_hyperplane = np.abs(detector.decision_function(test_embeddings))
        split_day = [split + "_day_" + str(day) for split, day in zip(splits, days)]

        df_svm = pd.DataFrame({
            "split_day": split_day,
            "label": labels,
            "preds": preds,
            "dist_from_hp": dist_from_hyperplane
        })

        anomaly_scores, labels, split_days = [], [], []
        # Calculate anomaly score for each pair (split, day_index)

        for s_d in np.unique(split_day):
            df_svm_filt = df_svm[df_svm['split_day'] == s_d]
            dist_from_hp = df_svm_filt[df_svm_filt['preds'] == -1]["dist_from_hp"]

            score = svm_score(df_svm_filt['preds'], dist_from_hp.abs())

            anomaly_scores.append(score)
            labels.append(df_svm_filt['label'].iloc[0])
            split_days.append(s_d)

        return {
            "anomaly_scores": np.array(anomaly_scores),
            "labels": np.array(labels, dtype=np.int64),
            "split_days": split_days
        }


def classification_train_loop(train_dset, val_dset, model, epochs, batch_size, patience, learning_rate, pt_file, device,
                              num_workers):

    # Initialize dataloaders & optimizers
    model = model.to(device)
    train_dloader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dloader = DataLoader(val_dset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    optim = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    lr_scheduler = CosineAnnealingLR(optimizer=optim, T_max=epochs, eta_min=1e-6)
    ear_stopping = EarlyStopping(patience=patience, verbose=True, path=pt_file)
    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(device=device)

    train_loss, val_loss, best_val_loss = 0., 0., np.inf
    _padding = len(str(epochs + 1))

    for epoch in range(1, epochs + 1):
        model.train()
        with tqdm(train_dloader, unit="batch", leave=False, desc="Training set") as tbatch:
            for d in tbatch:
                # Forward
                org_features, labels = d["features"], d["patient_id"] - 1  # Adjust labels
                org_features, labels = org_features.to(device), labels.to(device)
                out, _ = model(org_features)

                loss = loss_fn(out, labels)
                train_loss += loss.item()

                # Backward
                optim.zero_grad()
                loss.backward()
                optim.step()
        train_loss /= len(train_dloader)
        lr_scheduler.step()

        # Validation loop
        model.eval()
        y_trues, y_preds = [], []
        with torch.no_grad():
            with tqdm(val_dloader, unit="batch", leave=False, desc="Validation set") as vbatch:
                for d in vbatch:
                    # Forward
                    org_features, labels = d["features"], d['patient_id'] - 1
                    org_features, labels = org_features.to(device), labels.to(device)
                    out, _ = model(org_features)

                    loss = loss_fn(out, labels)
                    val_loss += loss.item()
                    preds = out.argmax(axis=1)

                    # Gather all for epoch results
                    y_trues.append(labels.cpu().numpy())
                    y_preds.append(preds.cpu().numpy())
        val_loss /= len(val_dloader)
        y_trues, y_preds = np.concatenate(y_trues), np.concatenate(y_preds)

        f1, acc = f1_score(y_true=y_trues, y_pred=y_preds, average='macro'), accuracy_score(y_true=y_trues,
                                                                                            y_pred=y_preds)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_loss_acc = acc
            best_val_loss_f1 = f1

        print(f"Epoch {epoch:<{_padding}}/{epochs}. Train Loss: {train_loss:.3f}. Val Loss: {val_loss:.3f}.", end=" ")
        print(f"F1: {f1:.3f}. Acc: {acc:.3f}.")

        ear_stopping(val_loss, model, epoch)
        if ear_stopping.early_stop:
            print("Early Stopping.")
            return best_val_loss
        train_loss, val_loss = 0.0, 0.0

    return {"val_loss": best_val_loss, "f1": best_val_loss_f1, "acc": best_val_loss_acc}


def validate_classification(train_dset, val_dset, model, batch_size, device, num_workers):
    model = model.to(device)
    train_dloader = DataLoader(train_dset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_dloader = DataLoader(val_dset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Get embeddings from train set
    train_embeddings = []
    model.eval()
    with torch.no_grad():
        with tqdm(train_dloader, unit="batch", desc="Training set", leave=False) as tbatch:
            for d in tbatch:
                org_features = d['features']
                org_features = org_features.to(device)

                out, emb = model(org_features)
                train_embeddings.append(emb.cpu())

    # Fit SVM Detector
    train_embeddings = torch.cat(train_embeddings).numpy()
    detector = OneClassSVM()
    scaler = StandardScaler()
    scaler.fit(train_embeddings)
    train_embeddings = scaler.transform(train_embeddings)
    detector.fit(train_embeddings)

    # Gather test data
    test_embeddings, posteriors, labels, splits, days = [], [], [], [], []

    model.eval()
    with torch.no_grad():
        with tqdm(val_dloader, unit="batch", desc="Valdation", leave=False) as vbatch:
            for d in vbatch:
                org_features, patient_id = d['features'], d['patient_id']
                org_features, patient_id = org_features.to(device), patient_id.to(device)

                out, emb = model(org_features)
                test_embeddings.append(emb.cpu())
                out = TF.softmax(out, dim=1)
                posteriors.append(out[:, patient_id[0] - 1].cpu())
                labels.append(d['label'].numpy())
                splits += d['split']
                days.append(d['day_index'].numpy())

            test_embeddings = torch.cat(test_embeddings).numpy()
            posteriors = torch.cat(posteriors).numpy()
            labels = np.concatenate(labels)
            days = np.concatenate(days)

    test_embeddings = scaler.transform(test_embeddings)
    anomalies = detector.predict(test_embeddings)

    df = pd.DataFrame({
        "split": splits,
        "day_index": days,
        "anomalies": anomalies,
        "posteriors": posteriors,
        "label": labels
    })

    f_splits, f_days, f_anomalies, f_posteriors, f_labels = [], [], [], [], []

    split_days = np.unique([s + '-' + str(d) for s, d in zip(df['split'], df['day_index'])])

    for s_d in split_days:
        s, d = s_d.split("-")[0], int(s_d.split("-")[1])
        df_f = df[(df['split'] == s) & (df['day_index'] == d)]
        f_splits.append(s)
        f_days.append(d)

        f_posteriors.append(1 - df_f['posteriors'].min())
        f_anomalies.append(df_f[df_f['anomalies'] == -1].shape[0] / df_f.shape[0])
        f_labels.append(df_f['label'].iloc[0])

    return pd.DataFrame({
        "split": f_splits,
        "day_index": f_days,
        "anomaly_scores_posteriors": f_posteriors,
        "anomaly_scores_svm": f_anomalies,
        "label": f_labels
    })
