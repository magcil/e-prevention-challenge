import os
import sys
from typing import Dict, List, Optional
import random

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import torch.nn as nn
from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd

from utils.parse import get_features, get_relapses
from preprocess.config import FEATURE_NAMES


class PatientDataset(Dataset):

    def __init__(self,
                 track_id: Optional[int],
                 patient_id: Optional[int],
                 mode: Optional[str],
                 window_size: int = 64,
                 extension: str = ".parquet",
                 feature_mapping: Dict[str, List[str]] = FEATURE_NAMES,
                 patient_features: Optional[List[str]] = None,
                 from_path: bool = True):

        self.extension = extension
        self.mode = mode
        self.track_id = track_id
        self.patient_id = patient_id

        if from_path:
            self.patient_features = get_features(track_id=track_id,
                                                 patient_id=patient_id,
                                                 mode=mode,
                                                 extension=extension)
        else:
            self.patient_features = patient_features

        self.window_size = window_size

        self.columns_to_keep = []
        # Get columns corresponding to the features to keep
        for k, v in feature_mapping.items():
            self.columns_to_keep += v

        self.mean, self.std = np.zeros((len(self.columns_to_keep), 1), dtype=np.float32), np.ones(
            (len(self.columns_to_keep), 1), dtype=np.float32)

    def __len__(self):
        return len(self.patient_features)

    def __getitem__(self, index):

        # Get features
        features = self._parse_data(self.patient_features[index])
        split, day_index = self._get_file_info(self.patient_features[index])

        # Get label if mode on val
        if self.mode == "val":
            label = self._get_label(split=split, day_index=day_index)
        else:
            label = 0

        # Initialize mask
        mask = np.ones(shape=(features.shape[0], self.window_size), dtype=np.float32)

        # Check if there are enough 5Min intervals
        if features.shape[1] >= self.window_size:
            # Sample random intervals
            time_indices = sorted(random.sample(range(features.shape[1]), self.window_size))
            features = features[:, time_indices]
        else:
            # Get all features and append zeros
            difference = self.window_size - features.shape[1]
            old_dim = features.shape[1]
            features = np.concatenate([features, np.zeros(shape=(features.shape[0], difference))], axis=1)
            mask[:, old_dim:] = 0.

        # Get nan/inf values and mask them too
        mask[np.isnan(features) | np.isinf(features)] = 0.
        features[np.isnan(features) | np.isinf(features)] = 0.
        # Apply normalization / Broadcasting
        features = (features - self.mean) / self.std
        features = features.astype(np.float32)

        # Apply normalization and return Dict
        return {
            "features": torch.unsqueeze(torch.from_numpy(features), dim=0),
            "mask": torch.unsqueeze(torch.from_numpy(mask), dim=0),
            "split": split,
            "day_index": day_index,
            "label": label
        }

    def _parse_data(self, file: str) -> pd.DataFrame:
        # Parse data
        if self.extension == ".parquet":
            df = pd.read_parquet(file, engine="fastparquet")
        elif self.extension == ".csv":
            df = pd.read_csv(file, parse_dates=["DateTime"])

        # Get features
        features = df[self.columns_to_keep].to_numpy(dtype=np.float32)

        # T x F -> F x T (F: features, T: time)
        features = features.T

        return features

    def _get_file_info(self, file: str):
        f = file.split("/")
        split, f = f[-3], f[-1]
        day_index = int(f.split("_")[1][:2])

        return split, day_index

    def _cal_statistics(self):
        # Get mean and std from all features / Use only on train
        all_features = []
        for file in self.patient_features:
            f = self._parse_data(file)
            all_features.append(f)

        all_features = np.concatenate(all_features, axis=1)
        # Replace inf values with nans
        all_features[np.isinf(all_features)] = np.nan
        # Use nan aggregations
        self.mean, self.std = np.nanmean(all_features, axis=1), np.nanstd(all_features, axis=1)
        self.mean = self.mean.reshape(*self.mean.shape, 1)
        self.std = self.std.reshape(*self.std.shape, 1)

    def _upsample_data(self, upsample_size: int):
        upsample_list = []
        for file in self.patient_features:
            upsample_list += [file] * upsample_size

        self.patient_features = upsample_list

    def _get_label(self, split, day_index):
        num = int(split.split("_")[1])
        relapses = get_relapses(track=self.track_id, patient=self.patient_id, num=num, mode=self.mode)

        return relapses[relapses['day_index'] == day_index]['relapse'].item()
