# Creating a custom dataset for reading the dataframe and loading it into the dataloader to pass it to the neural network at a later stage for finetuning the model and to prepare it for predictions
import torch

import os
import numpy as np
import pandas as pd

from utils.normalize import normalize_cols

pd.options.mode.chained_assignment = None


class RelapseDetectionDataset(torch.utils.data.Dataset):

    def __init__(self,
                 features_paths,
                 patient_dir,
                 window_size,
                 spd,
                 train_mean: dict,
                 train_std: dict,
                 split: str,
                 state='',
                 calc_norm=False,
                 format = 'parquet'):

        self.features_paths = features_paths
        self.patient_dir = patient_dir
        self.window_size = window_size
        self.split = split
        self.state = state
        self.spd = spd
        self.format = format

        # Normalize dataframe columns
        print(f'Normalizing the {self.split} DataFrame columns. This might take some seconds...')

        self.norm_save_dir = self.patient_dir + f'/norm_features_{self.split}_{state}'

        if calc_norm:
            means, stds = normalize_cols(self.features_paths, self.norm_save_dir, None, None, calc_norm, format=self.format)
            self.means = means
            self.stds = stds
        else:
            _, _ = normalize_cols(self.features_paths, self.norm_save_dir, train_mean, train_std, format=self.format)

        self.normalized_feat_paths = [item for item in os.listdir(self.norm_save_dir) for _ in range(self.spd)]

        self.ordered_columns = [
            'DateTime', 'heartRate_nanmean', 'rRInterval_nanmean', 'rRInterval_rmssd', 'rRInterval_sdnn', "gyr_mean",
            "gyr_std", "gyr_delta_mean", "gyr_delta_std", "acc_mean", "acc_std", "acc_delta_std", "interval_sleep",
            "aggr_sleep", "n_sleep", "cos_t", "sin_t"
        ]  # specifying a standard column order

    def __len__(self):
        return len(self.normalized_feat_paths)

    def __getitem__(self, index):

        location = self.normalized_feat_paths[index]
        if self.format == 'parquet':
            day_df = pd.read_parquet(os.path.join(self.norm_save_dir, location), engine='fastparquet')
        elif self.format == 'csv':
            day_df = pd.read_csv(os.path.join(self.norm_save_dir, location), parse_dates=["DateTime"])
        day_df.sort_values(by="DateTime", inplace=True, ascending=True)
        day_df = day_df[self.ordered_columns]  # forcing the column order as specified above

        # drop DateTime column
        fragment = day_df.drop(columns=["DateTime"])

        mask = np.ones(shape=(self.window_size, len(self.ordered_columns) - 1), dtype=np.float32)

        if len(day_df) - self.window_size > 0:
            fragment = fragment.sample(n=self.window_size).sort_index()
            rows = [row.to_list() for index, row in fragment.iterrows()]
            features = np.array(rows, dtype=np.float32)
            # Get nan/inf values and mask them too
            mask[np.isnan(features) | np.isinf(features)] = 0.
            features[np.isnan(features) | np.isinf(features)] = 0.

        else:  # zero-padding
            difference = self.window_size - len(fragment)
            rows = [row.to_list() for index, row in fragment.iterrows()]

            for i in range(difference):
                rows.append([0 for i in range(len(fragment.columns))])

            features = np.array(rows, dtype=np.float32)
            # Get nan/inf values and mask them too
            mask[np.isnan(features) | np.isinf(features)] = 0.
            features[np.isnan(features) | np.isinf(features)] = 0.
            # Mask padding also
            mask[len(fragment):, :] = 0.

        return torch.unsqueeze(torch.from_numpy(features), dim=0), torch.unsqueeze(torch.from_numpy(mask), dim=0)
