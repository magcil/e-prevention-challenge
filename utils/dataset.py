# Creating a custom dataset for reading the dataframe and loading it into the dataloader to pass it to the neural network at a later stage for finetuning the model and to prepare it for predictions
import torch
from torchvision import transforms
import os
import pandas as pd
import math
import random
from utils.normalize import normalize_cols

pd.options.mode.chained_assignment = None


class RelapseDetectionDataset(torch.utils.data.Dataset):

    def __init__(self, features_paths, patient_dir, window_size, spd, split:str, state=''):
        
        self.features_paths = features_paths
        self.patient_dir = patient_dir
        self.window_size = window_size
        self.split = split
        self.spd = spd

        # Normalize dataframe columns
        print(f'Normalizing the {self.split} DataFrame columns. This might take some seconds...')
        
        self.norm_save_dir = self.patient_dir + f'/norm_features_{self.split}_{state}'
        normalize_cols(self.features_paths, self.split, self.norm_save_dir)
        #self.normalized_feat_paths = self.spd * os.listdir(self.norm_save_dir)
        self.normalized_feat_paths = [item for item in os.listdir(self.norm_save_dir) for _ in range(self.spd)]

        #print(f'{split} normalized feat paths:', self.normalized_feat_paths)

        
        
        self.ordered_columns = ['DateTime', 'heartRate_nanmean', 'rRInterval_nanmean', 'rRInterval_rmssd', 'rRInterval_sdnn',
       'rRInterval_lombscargle_power_high', 'gyr_mean', 'gyr_std', 'gyr_delta_mean', 'gyr_delta_std', 'acc_mean', 'acc_std',
       'acc_delta_mean', 'acc_delta_std', 'sin_t', 'cos_t'] # specifying a standard column order
        


    def __len__(self):
        return len(self.normalized_feat_paths)

    

    def __getitem__(self, index):
        
        location = self.normalized_feat_paths[index]

        day_df = pd.read_parquet(os.path.join(self.norm_save_dir, location), engine='fastparquet')
        day_df = day_df[self.ordered_columns] # forcing the column order as specified above

        #print('day df columns:', day_df.columns)


        # drop DateTime column
        fragment = day_df.drop(columns=["DateTime"])


        if len(day_df) - self.window_size > 0:
            start_ = random.randint(0, len(day_df)-self.window_size)
            fragment = fragment.sample(n=self.window_size).sort_index()
            rows = [row.to_list() for index, row in fragment.iterrows()]

        else: # zero-padding
            difference = self.window_size - len(fragment)
            rows = [row.to_list() for index, row in fragment.iterrows()]

            for i in range(difference):
                rows.append([0 for i in range(len(fragment.columns))])
        
        item = torch.Tensor(rows).unsqueeze(0)

        return item, location