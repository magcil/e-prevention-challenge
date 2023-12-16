import os
import sys
import argparse
import json
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import torch
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import pandas as pd
import numpy as np

from models.classifiers import CNN_Classifier
import utils.parse as parser
from datasets.dataset import PatientDataset
from training.torch_utils import combine_patient_datasets
from training.loops import classification_train_loop, validate_classification


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-', '--config', required=True, help='Classification training config json file.')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    with open(args.config, "r") as f:
        json_config = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = {
        "Patient_id": [],
        "ROC AUC (posteriors)": [],
        "PR AUC (posteriors)": [],
        "ROC AUC (SVM)": [],
        "PR AUC (SVM)": []
    }

    # Combine and split patient datasets
    train_dset, val_dset = combine_patient_datasets(track_id=json_config['track_id'],
                                                    patients=json_config['patients'],
                                                    file_format=json_config['file_format'],
                                                    train_ratio=json_config['split_ratio'],
                                                    window_size=json_config['window_size'],
                                                    feature_mapping=json_config['feature_mapping'],
                                                    upsampling_size=json_config['upsampling_size'])

    # Initialize Model
    model = CNN_Classifier(H=json_config['Height'], W=json_config['Width'], num_classes=len(json_config['patients']))
    model = model.to(device)
    pt_file = os.path.join(json_config['pretrained_models'], "classifier_" + str(datetime.today().date()) + ".pt")

    # Train
    train_results = classification_train_loop(train_dset=train_dset,
                                              val_dset=val_dset,
                                              model=model,
                                              epochs=json_config['epochs'],
                                              batch_size=json_config['batch_size'],
                                              patience=json_config['patience'],
                                              learning_rate=json_config['learning_rate'],
                                              pt_file=pt_file,
                                              device=device,
                                              num_workers=json_config['num_workers'])
    print(f"Train Results: {train_results}")

    # Load Best model
    model.load_state_dict(torch.load(pt_file))
    # Validate each patient
    for patient_id in json_config['patients']:
        train_dset = PatientDataset(track_id=json_config['track_id'],
                                    patient_id=patient_id,
                                    mode="train",
                                    window_size=json_config['window_size'],
                                    extension=json_config['file_format'],
                                    feature_mapping=json_config['feature_mapping'])

        val_dset = PatientDataset(track_id=json_config['track_id'],
                                  patient_id=patient_id,
                                  mode="val",
                                  window_size=json_config['window_size'],
                                  extension=json_config['file_format'],
                                  feature_mapping=json_config['feature_mapping'])
        train_dset._cal_statistics()

        train_dset._upsample_data(upsample_size=json_config['upsampling_size'])
        val_dset._upsample_data(upsample_size=json_config['upsampling_size'])

        df = validate_classification(train_dset=train_dset,
                                     val_dset=val_dset,
                                     model=model,
                                     batch_size=64,
                                     device=device,
                                     num_workers=json_config['num_workers'])

        precision, recall, _ = precision_recall_curve(y_true=df['label'], probas_pred=df['anomaly_scores_posteriors'])
        fpr, tpr, _ = roc_curve(df['label'], df['anomaly_scores_posteriors'])
        results['Patient_id'].append(patient_id)
        results['ROC AUC (posteriors)'].append(auc(fpr, tpr))
        results['PR AUC (posteriors)'].append(auc(recall, precision))

        precision, recall, _ = precision_recall_curve(df['label'], df['anomaly_scores_svm'])
        fpr, tpr, _ = roc_curve(df['label'], df['anomaly_scores_svm'])

        results['ROC AUC (SVM)'].append(auc(fpr, tpr))
        results['PR AUC (SVM)'].append(auc(recall, precision))

        # Write csvs
        final_df = pd.DataFrame(results)
        final_df.to_csv("classification_results_" + str(datetime.today().date()) + ".csv")

        splits = np.unique(df['split'])

        for sp in splits:
            dff = df[df['split'] == sp]
            s, n = sp.split("_")[0], int(sp.split("_")[1])

            to_path = parser.get_path(track=json_config['track_id'], patient=patient_id, mode=s, num=n)
            dff.sort_values(by="day_index")
            dff.to_csv(os.path.join(to_path, f"classification_results_{datetime.today().date()}.csv"))
