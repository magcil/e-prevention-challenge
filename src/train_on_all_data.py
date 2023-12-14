import os
import sys
import argparse
import json
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset

from end_to_end_evaluation import train_loop
import utils.parse as parser
from datasets.dataset import PatientDataset
from models.convolutional_autoencoder import Autoencoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='Training config file.')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    with open(args.config, "r") as f:
        json_config = json.load(f)
    track_id = json_config["track_id"]
    patients = json_config["patients"]
    feature_mapping = json_config["feature_mapping"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prepare datasets
    train_dset, val_dset = [], []

    for patient_id in patients:
        X = parser.get_features(track_id=track_id,
                                patient_id=patient_id,
                                mode="train",
                                extension=json_config["file_format"])
        X_train, X_val = train_test_split(X, test_size=1 - json_config['train_ratio'], random_state=42)
        # Initialize / Upsample / append
        patient_train_dset = PatientDataset(track_id=track_id,
                                            patient_id=patient_id,
                                            mode="train",
                                            window_size=json_config["window_size"],
                                            extension=json_config["file_format"],
                                            feature_mapping=feature_mapping,
                                            patient_features=X_train,
                                            from_path=False)

        patient_val_dset = PatientDataset(track_id=track_id,
                                          patient_id=patient_id,
                                          mode="train",
                                          window_size=json_config["window_size"],
                                          extension=json_config["file_format"],
                                          feature_mapping=feature_mapping,
                                          patient_features=X_val,
                                          from_path=False)
        # Calculate Standardization statistics
        patient_train_dset._cal_statistics()
        patient_val_dset.mean, patient_val_dset.std = patient_train_dset.mean, patient_train_dset.std

        # Upsample
        patient_train_dset._upsample_data(upsample_size=json_config["upsampling_size"])
        patient_val_dset._upsample_data(upsample_size=json_config["upsampling_size"])

        # Append
        train_dset.append(patient_train_dset)
        val_dset.append(patient_val_dset)

    # Concat datasets
    train_dset, val_dset = ConcatDataset(train_dset), ConcatDataset(val_dset)

    # Initialize model
    model = Autoencoder()
    model = model.to(device)

    pt_file = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
                           json_config["pretrained_models"],
                           f"track_{track_id}_on_all_data_{datetime.today().date()}.pt")

    # Start training
    train_loop(train_dset=train_dset,
               val_dset=val_dset,
               model=model,
               epochs=json_config["epochs"],
               batch_size=json_config["batch_size"],
               patience=json_config["patience"],
               learning_rate=json_config["learning_rate"],
               pt_file=pt_file,
               device=device,
               num_workers=json_config["num_workers"])
