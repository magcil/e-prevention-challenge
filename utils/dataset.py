# Creating a custom dataset for reading the dataframe and loading it into the dataloader to pass it to the neural network at a later stage for finetuning the model and to prepare it for predictions
import torch
from torchvision import transforms
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor, PILToTensor
from PIL import Image
import os
import pandas as pd
import math
import random
from utils.normalize import normalize_cols

from sklearn.preprocessing import MinMaxScaler
pd.options.mode.chained_assignment = None


class RelapseDetectionDataset(torch.utils.data.Dataset):

    def __init__(self, base_path, window_size, split:str):
        
        self.base_path = base_path
        self.window_size = window_size
        self.split = split

        self.map = {
            '0': 'normal',
            '1': 'relapse'
        }

        if isinstance(self.base_path, list):
            if len(self.base_path)>1:
                self.features_paths = list()
                for path in self.base_path:
                    self.features_paths.append(os.listdir(path))
            else:
                self.features_paths = os.listdir(self.base_path[0])
        else:
            self.features_paths = os.listdir(self.base_path)
        

        if self.split=='train':
            for i in range(len(self.features_paths)):
                print('len of split:', math.floor(0.8 * (len(self.features_paths[i]))))
                self.features_paths[i] = self.features_paths[i][:math.floor(0.8 * (len(self.features_paths[i])))]
        elif self.split == 'validation':
            for i in range(len(self.features_paths)):
                self.features_paths[i] = self.features_paths[i][math.floor(0.8 * (len(self.features_paths[i]))):]


        # Normalize dataframe columns
        print(f'Normalizing the {self.split} DataFrame columns. This might take some seconds...')
        
        if isinstance(self.base_path, list):
            for i, path in enumerate(self.base_path):
                self.norm_base_path = os.path.dirname(os.path.dirname(path)) + f'/norm_features_{self.split}'
                normalize_cols(path, self.features_paths[i], self.norm_base_path, i, self.split)
                self.normalized_feat_paths = os.listdir(self.norm_base_path)
        else:
            state = self.map[os.path.dirname(self.base_path)[-1]]
            self.norm_base_path = os.path.dirname(os.path.dirname(self.base_path)) + f'/norm_features_{self.split}_{state}'
            normalize_cols(self.base_path, self.features_paths, self.norm_base_path, state, self.split)
            self.normalized_feat_paths = os.listdir(self.norm_base_path)
        


    def __len__(self):
        return len(self.normalized_feat_paths)

    

    def __getitem__(self, index):
        
        location = os.path.join(self.norm_base_path, self.normalized_feat_paths[index])
        day_df = pd.read_parquet(os.path.join(self.norm_base_path, self.normalized_feat_paths[index]), engine='fastparquet')

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