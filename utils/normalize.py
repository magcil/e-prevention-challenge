import pandas as pd
import os
import numpy as np


def normalize_cols(features_paths, split, save_dir):
    TIME_RELATED = ['DateTime']
    for f in features_paths:
        day_df = pd.read_parquet(f, engine='fastparquet')

        for col in day_df.columns:
            if col not in TIME_RELATED:
                day_df[col].replace([-np.inf, np.inf], [np.nan, np.nan], inplace=True)

                if day_df[col].isnull().values.any() == True:
                        day_df[col].fillna(day_df[col].mean(), inplace=True)
                
                day_df[col] = (day_df[col] - day_df[col].min()) / (day_df[col].max() - day_df[col].min())
        
        filename_chunks = f.split('/')
        idx, prq = filename_chunks[-3], filename_chunks[-1]
        filename = f'norm_{idx}_{prq}'

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        out_file = os.path.join(save_dir, filename)
        day_df.to_parquet(out_file, engine='fastparquet')