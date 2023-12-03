import pandas as pd
import numpy as np
import torch
import math
import os

def split_train_val(train_paths):

    # train & val splitting
    train_features_paths = list()
    for path in train_paths:
        for filename in os.listdir(path):
            train_features_paths.append(path + f'/{filename}')

    train_val = torch.utils.data.random_split(train_features_paths, [math.floor(0.8 * len(train_features_paths)), math.ceil(0.2 * len(train_features_paths))], generator=torch.Generator().manual_seed(42))

    train, val = list(train_val[0]), list(train_val[1])

    return train, val


def handle_dev(dev_paths):

    # dev/test handling
    test_features_normal_paths, test_features_relapse_paths = list(), list()
            
    for path in dev_paths:
        relapses = pd.read_csv(os.path.join(os.path.dirname(path), 'relapses.csv'), header=0, names=['relapse', 'day_index'])
        for filename in os.listdir(path):
            day_idx = int(filename.split('.')[0].split('_')[1])
            if relapses.loc[relapses["day_index"]==day_idx]["relapse"].values[0] == 0:
               test_features_normal_paths.append(path + f'/{filename}')
                
            else:
                test_features_relapse_paths.append(path + f'/{filename}')
        
    print('TEST FEATURES NORMAL:', test_features_normal_paths)
    print('TEST FEATURES RELAPSED:', test_features_relapse_paths)

    dev_features_paths = [test_features_normal_paths, test_features_relapse_paths]
    
    return dev_features_paths