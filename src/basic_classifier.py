"""
A script to train/test a basic one-class svm classifier,
using a mean feature vector per sample (same size to the number of features).


The only step required before
running this script is to have created the .parquet files of the initial features
(produced by preprocess/baseline_features_extraction or preprocess/extract_all_features
"""


import os
import sys
sys.path.insert(1, os.path.join(os.path.dirname(sys.path[0])))


import torch
import numpy as np
from sklearn.svm import OneClassSVM
import pickle
from sklearn.metrics import accuracy_score, precision_recall_curve, roc_curve, auc
import argparse
from collections import Counter
from sklearn.model_selection import train_test_split
import utils.parse as parser
from datasets.dataset import PatientDataset
import pandas as pd
from datetime import datetime
import json
import pdb
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='Json config file.')

    return parser.parse_args()
def majority_vote(lst):
    lst = Counter(list(lst))
    return lst.most_common(1)[0][0]

def load_data(paths, track_id, patient_id, mode, window_size, feature_mapping):
    if  mode=="train":
        dataset = PatientDataset(track_id=track_id,
                                patient_id=patient_id,
                                mode=mode,
                                window_size=window_size,
                                extension=json_config['file_format'],
                                feature_mapping=feature_mapping,
                                patient_features=paths,
                                from_path=False)
    else:
        dataset = PatientDataset(track_id=track_id,
                               patient_id=patient_id,
                               mode= mode,
                               window_size=window_size,
                               extension=json_config["file_format"],
                               feature_mapping=feature_mapping,
                               from_path=True)

    return dataset
def raw_features(dataset):
    X = []
    days = []
    labels = []
    splits = []
    for i in range(len(dataset)):
        features, mask = dataset[i]["features"], dataset[i]["mask"]
        day_mean_feature_vector = torch.mean(features, dim=2).squeeze()
        X.append(day_mean_feature_vector)
        days.append(dataset[i]['day_index'])
        labels.append(dataset[i]['label'])
        splits.append(dataset[i]['split'])
    X = np.array(X)
    df = pd.DataFrame({"day_index": days, "label": labels, "split": splits})
    return X, df

def train(checkpoint, track_id, patient_id, window_size, upsampling_size, feature_mapping):
    X = parser.get_features(track_id=track_id, patient_id=patient_id, mode="train")
    train_paths, val_paths = train_test_split(X, test_size=1 - json_config['split_ratio'], random_state=42)
    train_dataset = load_data(train_paths, track_id, patient_id, "train", window_size, feature_mapping)
    # Calculate statistics (mean, std) on train and pass to other datasets
    train_dataset._cal_statistics()
    mu, std = train_dataset.mean, train_dataset.std
    val_dset = load_data(val_paths, track_id, patient_id, "train", window_size, feature_mapping)
    val_dset.mean, val_dset.std = mu, std
    # Upsample train/val sets
    train_dataset._upsample_data(upsample_size=upsampling_size)
    val_dset._upsample_data(upsample_size=upsampling_size)

    X_train, df_train = raw_features(train_dataset)
    X_val, df_val = raw_features(val_dset)

    # Train the One-Class SVM
    model = OneClassSVM(nu=0.05)  # We may need to tune the hyperparameter 'nu'
    model.fit(X_train)
    # Predict on the validation set
    df_val["y_val_pred"] = model.predict(X_val)
    preds_per_day = df_val.groupby(by=["split", "day_index"])["y_val_pred"].apply(majority_vote).reset_index()
    y_val_pred = list(preds_per_day["y_val_pred"])
    # Evaluate the model on the validation set
    accuracy = accuracy_score(np.ones_like(y_val_pred), y_val_pred)
    print(f"Validation Accuracy: {accuracy}")

    # Save the model to a file using pickle
    with open(checkpoint, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to {checkpoint}")

    output = open(checkpoint.split('.')[0] + '-means.pkl', 'wb')
    pickle.dump(mu, output)
    output.close()

    output = open(checkpoint.split('.')[0] + '-stds.pkl', 'wb')
    pickle.dump(std, output)
    output.close()


def test(model, track_id, patient_id, window_size, upsample_size, feature_mapping):
    with open(model, 'rb') as file:
        loaded_model = pickle.load(file)
    train_means_path = model.split('.')[0] + '-means.pkl'
    train_stds_path = model.split('.')[0] + '-stds.pkl'

    file_to_read = open(train_means_path, "rb")
    mu = pickle.load(file_to_read)

    file_to_read = open(train_stds_path, "rb")
    std = pickle.load(file_to_read)
    test_dset = load_data(None, track_id, patient_id, "val", window_size, feature_mapping)

    test_dset.mean, test_dset.std = mu, std
    test_dset._upsample_data(upsample_size=upsample_size)
    X_test, df_test = raw_features(test_dset)

    decisions = loaded_model.decision_function(X_test)
    decisions = list(map(lambda x: -x, decisions))
    df_test["y_test_pred"] = decisions

    preds_per_day = df_test.groupby(by=["split", "day_index"]).mean()

    preds_per_day['label'] = preds_per_day['label'].astype(int)
    min_of_mean = preds_per_day['y_test_pred'].min()
    max_of_mean = preds_per_day['y_test_pred'].max()
    preds_per_day['y_pred_mean_decisions'] = preds_per_day['y_test_pred'].map(lambda x: (x - min_of_mean) / (max_of_mean - min_of_mean))
    decisions = [(x - min(decisions)) / (max(decisions) - min(decisions)) for x in decisions]
    df_test["y_test_pred"] = decisions
    preds_per_day['y_pred_mean_proba'] = df_test.groupby(by=["split", "day_index"]).mean()["y_test_pred"]


    results = {}
    y_pred_mean_decisions = list(preds_per_day["y_pred_mean_decisions"])

    y_pred_mean_proba = list(preds_per_day['y_pred_mean_proba'])

    y_true = list(preds_per_day["label"])

    #ps_random = np.random.uniform(0, 1, len(y_pred_mean_decisions))
    ps_random = []
    for day in range(len(y_pred_mean_decisions)):
        ps_random.append(np.random.random(size=upsample_size).mean())
    ps_random = np.array(ps_random)

    # Evaluate the model on the potential anomalies

    # Compute metrics
    precision, recall, _ = precision_recall_curve(y_true, y_pred_mean_decisions)
    fpr, tpr, _ = roc_curve(y_true, y_pred_mean_decisions)
    results["Mean decisions"] = {}
    results["Mean decisions"]["ROC AUC"] = auc(fpr, tpr)
    results["Mean decisions"]['PR AUC'] = auc(recall, precision)

    precision, recall, _ = precision_recall_curve(y_true, y_pred_mean_proba)
    fpr, tpr, _ = roc_curve(y_true, y_pred_mean_proba)
    results["Mean probabilities"] = {}
    results["Mean probabilities"]["ROC AUC"] = auc(fpr, tpr)
    results["Mean probabilities"]['PR AUC'] = auc(recall, precision)

    # Compute metrics for random guess
    precision, recall, _ = precision_recall_curve(y_true, ps_random)
    fpr, tpr, _ = roc_curve(y_true, ps_random)
    results["Random"] = {}
    results["Random"]["ROC AUC"] = auc(fpr, tpr)
    results["Random"]['PR AUC'] = auc(recall, precision)
    results['Patient_id'] = patient_id
    results['Model'] = model

    # Write csvs
    final_df = pd.DataFrame(results)
    print(final_df)
    return final_df
    #final_df.to_csv("results_OneClassSVM_" + str(datetime.today()) + ".csv")

if __name__=='__main__':
    # Parse args
    args = parse_args()
    with open(args.config, "r") as f:
        json_config = json.load(f)

    feature_mapping = json_config["feature_mapping"]
    track_id = json_config["track_id"]
    patient = json_config["patient"]
    window_size = json_config["window_size"]
    checkpoint = json_config["checkpoint"]
    if_test = json_config["test"]

    if if_test:
        roc_dec, pr_dec, roc_prob, pr_prob, roc_random, pr_random = [], [], [], [], [], []
        all_runs_df = {}
        for i in range(10):
            final_df = test(checkpoint, track_id, patient, window_size, json_config["prediction_upsampling"], feature_mapping)
            roc_dec.append(final_df["Mean decisions"]["ROC AUC"])
            pr_dec.append(final_df["Mean decisions"]['PR AUC'])
            roc_prob.append(final_df["Mean probabilities"]["ROC AUC"])
            pr_prob.append(final_df["Mean probabilities"]['PR AUC'])
            roc_random.append(final_df["Random"]["ROC AUC"])
            pr_random.append(final_df["Random"]['PR AUC'])
        all_runs_df["Mean decisions"] = {}
        all_runs_df["Mean decisions"]["ROC AUC"] = np.average(roc_dec)
        all_runs_df["Mean decisions"]['PR AUC'] = np.average(pr_dec)
        all_runs_df["Mean decisions"]["ROC AUC std"] = np.std(roc_dec)
        all_runs_df["Mean decisions"]['PR AUC std'] = np.std(pr_dec)
        all_runs_df["Mean probabilities"] = {}
        all_runs_df["Mean probabilities"]["ROC AUC"] = np.average(roc_prob)
        all_runs_df["Mean probabilities"]['PR AUC'] = np.average(pr_prob)
        all_runs_df["Mean probabilities"]["ROC AUC std"] = np.std(roc_prob)
        all_runs_df["Mean probabilities"]['PR AUC std'] = np.std(pr_prob)
        all_runs_df["Random"] = {}
        all_runs_df["Random"]["ROC AUC"] = np.average(roc_random)
        all_runs_df["Random"]['PR AUC'] = np.average(pr_random)
        all_runs_df["Random"]["ROC AUC std"] = np.std(roc_random)
        all_runs_df["Random"]['PR AUC std'] = np.std(pr_random)
        all_runs_df = pd.DataFrame(all_runs_df)
        print(all_runs_df)
    else:
        train(checkpoint, track_id, patient, window_size, json_config["upsampling_size"], feature_mapping)

