import pandas as pd
import os
import numpy as np


def normalize_cols(features_paths, split, save_dir, mean, std, calc_norm=False):
    TIME_RELATED = ['DateTime']
    cols = ['heartRate_nanmean', 'rRInterval_nanmean', 'rRInterval_rmssd', 'rRInterval_sdnn',
       'rRInterval_lombscargle_power_high', 'gyr_mean', 'gyr_std', 'gyr_delta_mean', 'gyr_delta_std', 'acc_mean', 'acc_std',
       'acc_delta_mean', 'acc_delta_std', 'sin_t', 'cos_t']

    # ------------------------- Mean & std calculation ---------------------------------------
    if calc_norm:
        vals = dict()
        for col in cols:
            vals[col] = list()

        for f in features_paths:
            day_df = pd.read_parquet(f, engine='fastparquet')
            for col in day_df.columns:
                if col not in TIME_RELATED:
                    day_df[col].replace([-np.inf, np.inf], [np.nan, np.nan], inplace=True)
                    if day_df[col].isnull().values.any() == True:
                            day_df[col].fillna(day_df[col].mean(), inplace=True)
                    vals[col].extend(list(day_df[col]))

        means, stds = dict(), dict()
        for col in cols:
            means[col] = np.mean(vals[col])
            stds[col] = np.std(vals[col])

    for f in features_paths:
        day_df = pd.read_parquet(f, engine='fastparquet')

        for col in day_df.columns:
            if col in cols:
                day_df[col].replace([-np.inf, np.inf], [np.nan, np.nan], inplace=True)

                if day_df[col].isnull().values.any() == True:
                        day_df[col].fillna(day_df[col].mean(), inplace=True)

                if calc_norm:
                    day_df[col] = ((day_df[col] - means[col]) / stds[col])
                else:
                    day_df[col] = ((day_df[col] - mean[col]) / std[col])
        
        filename_chunks = f.split('/')
        idx, prq = filename_chunks[-3], filename_chunks[-1]
        filename = f'norm_{idx}_{prq}'

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        out_file = os.path.join(save_dir, filename)
        day_df.to_parquet(out_file, engine='fastparquet')
    
    if calc_norm:
        return means, stds
    else:
        return None, None