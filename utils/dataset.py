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

        if isinstance(self.base_path, list) and len(self.base_path)>1:
            multifile = True
            self.features_paths = list()
            for path in self.base_path:
                self.features_paths.extend(os.listdir(path))
        else:
            self.features_paths = os.listdir(self.base_path[0])
        
        print('self features paths:', self.features_paths)


        if split=='train':
            self.features_paths= self.features_paths[:math.floor(0.8 * (len(self.features_paths)))]
        elif split == 'validation':
            self.features_paths = self.features_paths[math.floor(0.8 * (len(self.features_paths))):]

        # Normalize dataframe columns
        print(f'Normalizing the {split} DataFrame columns. This might take some seconds...')
        
        self.norm_base_path = os.path.dirname(self.base_path[0]) + '/norm_features'
        normalize_cols(self.base_path[0], self.features_paths, self.norm_base_path)
        self.normalized_feat_paths = os.listdir(self.norm_base_path)
        


    def __len__(self):
        return len(self.features_paths)

    

    def __getitem__(self, index):
        
        day_df = pd.read_parquet(os.path.join(self.norm_base_path, self.normalized_feat_paths[index]), engine='fastparquet')

        # drop time-related columns (not sure if this is right)
        fragment = day_df.drop(columns=["DateTime", 'sin_t', 'cos_t'])


        if len(day_df) - self.window_size > 0:
            start_ = random.randint(0, len(day_df)-self.window_size)
            fragment = fragment.iloc[start_ : start_ + self.window_size]
            rows = [row.to_list() for index, row in fragment.iterrows()]

        else: # zero-padding
            difference = self.window_size - len(fragment)
            rows = [row.to_list() for index, row in fragment.iterrows()]

            for i in range(difference):
                rows.append([0 for i in range(len(fragment.columns))])
                
        

        flat_list = []
        for row in rows:
            flat_list.extend(row)
        
        item = torch.Tensor(flat_list).unsqueeze(1)

        
        return item