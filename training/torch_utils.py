import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset

import utils.parse as parser
from datasets.dataset import PatientDataset


def combine_patient_datasets(track_id, patients, file_format, train_ratio, window_size, feature_mapping,
                             upsampling_size):
    # Prepare datasets
    train_dset, val_dset = [], []

    for patient_id in patients:
        X = parser.get_features(track_id=track_id, patient_id=patient_id, mode="train", extension=file_format)
        X_train, X_val = train_test_split(X, test_size=1 - train_ratio, random_state=42)
        # Initialize / Upsample / append
        patient_train_dset = PatientDataset(track_id=track_id,
                                            patient_id=patient_id,
                                            mode="train",
                                            window_size=window_size,
                                            extension=file_format,
                                            feature_mapping=feature_mapping,
                                            patient_features=X_train,
                                            from_path=False)

        patient_val_dset = PatientDataset(track_id=track_id,
                                          patient_id=patient_id,
                                          mode="train",
                                          window_size=window_size,
                                          extension=file_format,
                                          feature_mapping=feature_mapping,
                                          patient_features=X_val,
                                          from_path=False)
        # Calculate Standardization statistics
        patient_train_dset._cal_statistics()
        patient_val_dset.mean, patient_val_dset.std = patient_train_dset.mean, patient_train_dset.std

        # Upsample
        patient_train_dset._upsample_data(upsample_size=upsampling_size)
        patient_val_dset._upsample_data(upsample_size=upsampling_size)

        # Append
        train_dset.append(patient_train_dset)
        val_dset.append(patient_val_dset)

    # Concat datasets
    train_dset, val_dset = ConcatDataset(train_dset), ConcatDataset(val_dset)
    
    return train_dset, val_dset
