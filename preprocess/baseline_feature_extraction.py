import os
import datetime
import pandas as pd
import numpy as np
import pyhrv
import scipy
import argparse
from typing import Dict, Optional
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
from utils.parse import parse_data, get_path, get_unique_days

valid_ranges = {
    "acc_X": (-19.6, 19.6),
    "acc_Y": (-19.6, 19.6),
    "acc_Z": (-19.6, 19.6),
    "gyr_X": (-573, 573),
    "gyr_Y": (-573, 573),
    "gyr_Z": (-573, 573),
    "heartRate": (0, 255),
    "rRInterval": (0, 2000),
}


def rmssd(x):
    x = x.dropna()
    try:
        rmssd = pyhrv.time_domain.rmssd(x)[0]
    except (ZeroDivisionError, ValueError):
        rmssd = np.nan

    return rmssd


def sdnn(x):
    x = x.dropna()
    try:
        sdnn = pyhrv.time_domain.sdnn(x)[0]
    except (ZeroDivisionError, ValueError):
        sdnn = np.nan

    return sdnn


def lombscargle_power_high(nni):
    # high frequencies
    l = 0.15 * np.pi / 2
    h = 0.4 * np.pi / 2
    freqs = np.linspace(l, h, 1000)
    hf_lsp = scipy.signal.lombscargle(nni.to_numpy(), nni.index.to_numpy(), freqs, normalize=True)
    return np.trapz(hf_lsp, freqs)


def get_norm(df):
    """ Returns the mean norm of the x,y,z columns of a dataframe"""
    df = df.dropna()
    return np.sqrt(df['X']**2 + df['Y']**2 + df['Z']**2).mean()


def time_encoding(slice):
    # Compute the sin and cos of timestamp (we have 12*24=288 5-minutes per day)
    mean_timestamp = slice['timecol'].astype('datetime64').mean()
    h = mean_timestamp.hour
    m = mean_timestamp.minute
    time_value = h * 60 + m
    sin_t = np.sin(time_value * (2. * np.pi / (60 * 24)))
    cos_t = np.cos(time_value * (2. * np.pi / (60 * 24)))
    return sin_t, cos_t


def extract_hr(df_hrm):
    # where hearRate is out of limits, set it to nan
    df_hrm.loc[df_hrm['heartRate'] <= valid_ranges['heartRate'][0], 'heartRate'] = np.nan
    df_hrm.loc[df_hrm['heartRate'] > valid_ranges['heartRate'][1], 'heartRate'] = np.nan

    # same for rRInterval
    df_hrm.loc[df_hrm['rRInterval'] <= valid_ranges['rRInterval'][0], 'rRInterval'] = np.nan
    df_hrm.loc[df_hrm['rRInterval'] > valid_ranges['rRInterval'][1], 'rRInterval'] = np.nan

    df_hrm = df_hrm.groupby(pd.Grouper(key='time', freq='5Min')).agg({
        'heartRate':
        np.nanmean,
        'rRInterval': [np.nanmean, rmssd, sdnn, lombscargle_power_high]
    })
    df_hrm.columns = df_hrm.columns.map('_'.join)

    # Get time encodings
    secs = df_hrm.index.to_series().dt.seconds
    h = secs // 3600
    m = (secs % 3600) // 60
    time_value = h * 60 + m
    df_hrm['sin_t'] = np.sin(time_value * (2 * np.pi / (60 * 24)))
    df_hrm['cos_t'] = np.cos(time_value * (2 * np.pi / (60 * 24)))

    return df_hrm


def extract_linacc(df_linacc):
    # where acc is out of limits, set it to nan
    df_linacc.loc[(df_linacc['X'] < valid_ranges['acc_X'][0]) | (df_linacc['X'] >= valid_ranges['acc_X'][1]),
                  'X'] = np.nan
    df_linacc.loc[(df_linacc['Y'] < valid_ranges['acc_Y'][0]) | (df_linacc['Y'] >= valid_ranges['acc_Y'][1]),
                  'Y'] = np.nan
    df_linacc.loc[(df_linacc['Z'] < valid_ranges['acc_Z'][0]) | (df_linacc['Z'] >= valid_ranges['acc_Z'][1]),
                  'Z'] = np.nan

    # Calculate Norm for each row / drop unecessary columns
    df_linacc['Norm'] = np.sqrt(df_linacc['X']**2 + df_linacc['Y']**2 + df_linacc['Z']**2)
    df_linacc.drop(columns=['X', 'Y', 'Z', 'day_index'], inplace=True)

    # Calculate 5Min Norm aggregations
    norm_aggrs = df_linacc.groupby(pd.Grouper(key='time', freq='5Min'))['Norm'].agg(acc_mean='mean', acc_std='std')

    # Calculate deltas
    deltas = df_linacc.groupby(pd.Grouper(key='time', freq='30S'))['Norm'].mean()
    deltas = deltas.diff() / deltas.index.to_series().diff().dt.total_seconds()
    deltas = deltas.resample("5Min").agg(acc_delta_mean='mean', acc_delta_std='std')

    return norm_aggrs.merge(deltas, left_index=True, right_index=True, how='inner')


def extract_sleep(df):
    return df


def extract_gyr(df):
    # where acc is out of limits, set it to nan
    df.loc[(df['X'] < valid_ranges['acc_X'][0]) | (df['X'] >= valid_ranges['acc_X'][1]), 'X'] = np.nan
    df.loc[(df['Y'] < valid_ranges['acc_Y'][0]) | (df['Y'] >= valid_ranges['acc_Y'][1]), 'Y'] = np.nan
    df.loc[(df['Z'] < valid_ranges['acc_Z'][0]) | (df['Z'] >= valid_ranges['acc_Z'][1]), 'Z'] = np.nan

    # Calculate Norm for each row / drop unecessary columns
    df['Norm'] = np.sqrt(df['X']**2 + df['Y']**2 + df['Z']**2)
    df.drop(columns=['X', 'Y', 'Z', 'day_index'], inplace=True)

    # Calculate 5Min Norm aggregations
    norm_aggrs = df.groupby(pd.Grouper(key='time', freq='5Min'))['Norm'].agg(gyr_mean='mean', gyr_std='std')

    # Calculate deltas
    deltas = df.groupby(pd.Grouper(key='time', freq='30S'))['Norm'].mean()
    deltas = deltas.diff() / deltas.index.to_series().diff().dt.total_seconds()
    deltas = deltas.resample("5Min").agg(gyr_delta_mean='mean', gyr_delta_std='std')

    return norm_aggrs.merge(deltas, left_index=True, right_index=True, how='inner')


def extract_step(df):
    return df


FEATURE_FUNC = {
    'gyr': extract_gyr,
    'hrm': extract_hr,
    'linacc': extract_linacc,
    'sleep': extract_sleep,
    'step': extract_step
}


def extract_day_features(df_dicts: Dict[str, pd.DataFrame]):
    """Extract features for a specific day
    
    Args:
        df_dicts (Dict[str, pd.DataFrame]): DataFrames filtered on specific day
    
    Returns:
        np.ndarray
    """
    features = []
    for dtype, df in df_dicts.items():
        feature_extractor = FEATURE_FUNC[dtype]
        df = feature_extractor(df)
        features.append(df.to_numpy())
    features = np.concatenate(features, axis=1)

    # Drop rows with NaN values
    return features[~np.isnan(features).any(axis=1)]


# function that does feature extraction for a patient
def extract_user_features(track: Optional[int] = None,
                          patient: Optional[int] = None,
                          mode: Optional[str] = None,
                          num: Optional[int] = None,
                          dtypes: Optional[list] = None):

    if 'all' in dtypes:
        dtypes = ['gyr', 'hrm', 'linacc', 'sleep', 'step']

    # Process each dataframe by filtering for each day
    # Get unique days
    days = get_unique_days(track, patient, mode, num)

    # Parse all dataframes
    full_dfs = {dtype: parse_data(track, patient, mode, num, dtype) for dtype in dtypes}

    # Create directory
    path_to_save = get_path(track, patient, mode, num) + "/features"
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    day = 0
    p_bar = tqdm(days,
                 desc=f'Extracting features for each day (Track: {track} | Patient: {patient} | Mode: {mode})',
                 leave=False,
                 postfix={"Day": f"{day} | {days[-1]}"})

    for day in p_bar:
        fil_dfs = {dtype: full_dfs[dtype][full_dfs[dtype]['day_index'] == day].copy(deep=True) for dtype in dtypes}
        day_features = extract_day_features(fil_dfs)
        out_file = path_to_save + f"/day_{day:02}"
        np.save(out_file, day_features)
        p_bar.set_postfix({"Day": f"{day} | {days[-1]}"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--track', type=int, required=True, choices=[1, 2], help='track 1 or 2')
    parser.add_argument('--patient', type=int, required=True, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9], help='which patient')
    parser.add_argument('--mode', type=str, required=True, choices=["train", "test", "val"], help='which split')
    parser.add_argument('--num', type=int, required=True, help='which number of split')
    parser.add_argument('--dtype', type=str, nargs='+', default=['all'])

    args = parser.parse_args()

    extract_user_features(args.track, args.patient, args.mode, args.num, args.dtype)
