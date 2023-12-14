import pandas as pd
import os
import numpy as np


def normalize_cols(features_paths, save_dir, mean, std, calc_norm=False, format = "parquet"):
    cols = [
        'heartRate_nanmean', 'rRInterval_nanmean', 'rRInterval_rmssd', 'rRInterval_sdnn', "gyr_mean", "gyr_std",
        "gyr_delta_mean", "gyr_delta_std", "acc_mean", "acc_std", "acc_delta_std", "interval_sleep", "aggr_sleep",
        "n_sleep", "cos_t", "sin_t"
    ]

    # ------------------------- Mean & std calculation ---------------------------------------
    if calc_norm:
        vals = dict()
        for col in cols:
            vals[col] = list()

        for f in features_paths:
            if format == 'parquet':
                day_df = pd.read_parquet(f, engine='fastparquet')
            elif format == 'csv':
                day_df = pd.read_csv(f, parse_dates=["DateTime"])
            for col in day_df.columns:
                if col in cols:
                    day_df[col].replace([-np.inf, np.inf], [np.nan, np.nan], inplace=True)
                    if day_df[col].isnull().values.any() == True:
                        day_df[col].fillna(day_df[col].mean(), inplace=True)
                    vals[col].extend(list(day_df[col]))

        means, stds = dict(), dict()
        for col in cols:
            means[col] = np.mean(vals[col])
            stds[col] = np.std(vals[col])
    # ----------------------------------------------------------------------------------------

    for f in features_paths:
        if format == 'parquet':
            day_df = pd.read_parquet(f, engine='fastparquet')
        elif format == 'csv':
            day_df = pd.read_csv(f, parse_dates=["DateTime"])

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
        if format == 'parquet':
            day_df.to_parquet(out_file, engine='fastparquet')
        elif format == 'csv':
            day_df.to_csv(out_file)

    if calc_norm:
        return means, stds
    else:
        return None, None
