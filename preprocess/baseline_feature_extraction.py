import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pyhrv
import scipy
import argparse
from typing import Dict, Optional, List
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


def get_std(df):
    df = df.dropna()
    return np.sqrt(df['X']**2 + df['Y']**2 + df['Z']**2).std()


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
    start_time = datetime.strptime('00:00:00', '%H:%M:%S').time()  # Start time
    start_time = datetime.combine(datetime.today(), start_time)

    if not df_hrm.empty:
        # where hearRate is out of limits, set it to nan
        df_hrm.loc[df_hrm['heartRate'] <= valid_ranges['heartRate'][0], 'heartRate'] = np.nan
        df_hrm.loc[df_hrm['heartRate'] > valid_ranges['heartRate'][1], 'heartRate'] = np.nan

        # same for rRInterval
        df_hrm.loc[df_hrm['rRInterval'] <= valid_ranges['rRInterval'][0], 'rRInterval'] = np.nan
        df_hrm.loc[df_hrm['rRInterval'] > valid_ranges['rRInterval'][1], 'rRInterval'] = np.nan

        # Convert Timedelta to datetime.time
        df_hrm['DateTime'] = df_hrm['time'].apply(lambda t: start_time + t)

        df_hrm = df_hrm.groupby(pd.Grouper(key='DateTime', freq='5Min')).agg({
            'heartRate':
            np.nanmean,
            'rRInterval': [np.nanmean, rmssd, sdnn, lombscargle_power_high]
        })
        df_hrm.columns = df_hrm.columns.map('_'.join)
        df_hrm.dropna(inplace=True)
    else:
        # Determine all 5-Min intervals in a day and fill with nans
        features = np.zeros(shape=(60 * 24 // 5, 4))
        features[:] = np.nan
        df_hrm = pd.DataFrame(features, index=pd.date_range(start_time, periods=60 * 24 // 5, freq="5Min"))
        df_hrm.columns = [
            'heartRate_nanmean', 'rRInterval_nanmean', 'rRInterval_rmssd', 'rRInterval_sdnn',
            'rRInterval_lombscargle_power_high'
        ]

    return df_hrm


def extract_linacc(df_linacc):
    start_time = datetime.strptime('00:00:00', '%H:%M:%S').time()  # Start time
    start_time = datetime.combine(datetime.today(), start_time)

    if not df_linacc.empty:
        # where acc is out of limits, set it to nan
        df_linacc.loc[(df_linacc['X'] < valid_ranges['acc_X'][0]) | (df_linacc['X'] >= valid_ranges['acc_X'][1]),
                      'X'] = np.nan
        df_linacc.loc[(df_linacc['Y'] < valid_ranges['acc_Y'][0]) | (df_linacc['Y'] >= valid_ranges['acc_Y'][1]),
                      'Y'] = np.nan
        df_linacc.loc[(df_linacc['Z'] < valid_ranges['acc_Z'][0]) | (df_linacc['Z'] >= valid_ranges['acc_Z'][1]),
                      'Z'] = np.nan

        # Convert Timedelta to datetime.time
        df_linacc['DateTime'] = df_linacc['time'].apply(lambda t: start_time + t)

        # Calculate 5Min Norm aggregations
        norm_aggrs = df_linacc.groupby(pd.Grouper(key='DateTime', freq='5Min')).apply(get_norm)
        norm_stds = df_linacc.groupby(pd.Grouper(key='DateTime', freq='5Min')).apply(get_std)

        # Calculate deltas
        deltas = df_linacc.groupby(pd.Grouper(key='DateTime', freq='30S')).apply(get_norm)
        deltas = deltas.diff() / deltas.index.to_series().diff().dt.total_seconds()
        deltas = deltas.resample("5Min").agg(['mean', 'std'])

        # Merge all and rename
        df_linacc = pd.concat([norm_aggrs, norm_stds, deltas], axis=1)
        df_linacc.dropna(inplace=True)
    else:
        # Determine all 5-Min intervals in a day and fill with nans
        features = np.zeros(shape=(60 * 24 // 5, 4))
        features[:] = np.nan
        df_linacc = pd.DataFrame(features, index=pd.date_range(start_time, periods=60 * 24 // 5, freq="5Min"))
    df_linacc.columns = ['acc_mean', 'acc_std', 'acc_delta_mean', 'acc_delta_std']

    return df_linacc


# Utility functions for sleep related features


# Filter sleep data on target day
def filter_sleep(df, day_index):
    return df[((df['start_date_index'] == day_index - 1) & (df['end_date_index'] == day_index)) |
              ((df['start_date_index'] == day_index) & (df['end_date_index'] == day_index)) |
              ((df['start_date_index'] == day_index) & (df['end_date_index'] == day_index + 1))].copy(deep=True)


def convert_to_datetime(df: pd.DataFrame, date, cols):
    df[cols] += datetime(year=date.year, month=date.month, day=date.day)
    return df


def adjust_days(df: pd.DataFrame, day_index: int):
    df.loc[df['start_date_index'] == day_index - 1, 'start_time'] -= timedelta(days=1)
    df.loc[df['end_date_index'] == day_index + 1, 'end_time'] += timedelta(days=1)
    return df


def append_sleep_features(f: pd.DataFrame, s: pd.DataFrame, day_index: int):
    """Append sleep features on the extracted features
    
    Args:
        f (pd.DataFrame): The extracted features
        s (pd.DataFrame): The sleep data
        day_index (int): The day index corresponding to features f
    Returns:
        pd.DataFrame: features + sleep features appended
    """
    date = f['DateTime'].iloc[0].date()
    s_filt = filter_sleep(s, day_index=day_index)

    s_filt = s_filt.pipe(convert_to_datetime, date=date, cols=['start_time', 'end_time']).pipe(adjust_days,
                                                                                               day_index=day_index)

    interval_sleeps, aggr_sleeps, n_sleeps = [], [], []

    for dtime in f['DateTime']:
        aggr_sleep = 0
        n_sleep = 0
        interval_sleep = 0
        for dt_times in s_filt.itertuples(index=False):
            start_time, end_time = dt_times.start_time, dt_times.end_time
            if start_time <= dtime <= end_time:
                # On sleeping interval
                interval_sleep = (dtime - start_time).total_seconds() // 60
                n_sleep += 1
            elif end_time < dtime:
                n_sleep += 1
                aggr_sleep += (end_time - start_time).total_seconds() // 60

        interval_sleeps.append(interval_sleep)
        aggr_sleeps.append(aggr_sleep + interval_sleep)
        n_sleeps.append(n_sleep)

    f['interval_sleep'] = interval_sleeps
    f['aggr_sleep'] = aggr_sleeps
    f['n_sleep'] = n_sleeps


def extract_gyr(df):

    start_time = datetime.strptime('00:00:00', '%H:%M:%S').time()  # Start time
    start_time = datetime.combine(datetime.today(), start_time)
    # Check if there are features in this day
    if not df.empty:
        # where acc is out of limits, set it to nan
        df.loc[(df['X'] < valid_ranges['gyr_X'][0]) | (df['X'] >= valid_ranges['gyr_X'][1]), 'X'] = np.nan
        df.loc[(df['Y'] < valid_ranges['gyr_Y'][0]) | (df['Y'] >= valid_ranges['gyr_Y'][1]), 'Y'] = np.nan
        df.loc[(df['Z'] < valid_ranges['gyr_Z'][0]) | (df['Z'] >= valid_ranges['gyr_Z'][1]), 'Z'] = np.nan

        # Convert Timedelta to datetime.time
        df['DateTime'] = df['time'].apply(lambda t: start_time + t)

        # Calculate 5Min Norm aggregations
        norm_aggrs = df.groupby(pd.Grouper(key='DateTime', freq='5Min')).apply(get_norm)
        norm_stds = df.groupby(pd.Grouper(key='DateTime', freq='5Min')).apply(get_std)

        # Calculate deltas
        deltas = df.groupby(pd.Grouper(key='DateTime', freq='30S')).apply(get_norm)
        deltas = deltas.diff() / deltas.index.to_series().diff().dt.total_seconds()
        deltas = deltas.resample("5Min").agg(['mean', 'std'])

        # Merge all and rename
        df = pd.concat([norm_aggrs, norm_stds, deltas], axis=1)
        df.dropna(inplace=True)
    # Else return nans - no values for that day
    else:
        # Determine all 5-Min intervals in a day and fill with nans
        features = np.zeros(shape=(60 * 24 // 5, 4))
        features[:] = np.nan
        df = pd.DataFrame(features, index=pd.date_range(start_time, periods=60 * 24 // 5, freq="5Min"))
    df.columns = ['gyr_mean', 'gyr_std', 'gyr_delta_mean', 'gyr_delta_std']

    return df


def extract_step(df):
    return df


FEATURE_FUNC = {
    'gyr': extract_gyr,
    'hrm': extract_hr,
    'linacc': extract_linacc,
    'sleep': append_sleep_features,
    'step': extract_step
}


def extract_day_features(df_dicts: Dict[str, pd.DataFrame], day_index: int):
    """Extract features for a specific day.
    
    Args:
        df_dicts (Dict[str, pd.DataFrame]): DataFrames containing raw data
        day_index (int): The index of day to filter data
    
    Returns:
        pd.DataFrame: Features of a specific day
    """
    dtypes = df_dicts.keys()
    fil_dfs = {
        dtype: df_dicts[dtype][df_dicts[dtype]['day_index'] == day_index].copy(deep=True)
        for dtype in dtypes if dtype not in ['sleep']
    }
    all_df = []
    for dtype, df in fil_dfs.items():
        if dtype not in ['sleep']:
            feature_extractor = FEATURE_FUNC[dtype]
            df = feature_extractor(df)
            all_df.append(df)

    # Combine all
    all_df = pd.concat(all_df, axis=1, join='inner')
    all_df = all_df.reset_index().rename(columns = {"index": "DateTime"})

    # Create time encodings
    h = all_df['DateTime'].dt.hour
    m = all_df['DateTime'].dt.minute
    time_value = h * 60 + m
    all_df['sin_t'] = np.sin(time_value * (2. * np.pi / (60 * 24)))
    all_df['cos_t'] = np.cos(time_value * (2. * np.pi / (60 * 24)))

    # Extract sleep features
    if 'sleep' in df_dicts.keys():
        FEATURE_FUNC['sleep'](all_df, df_dicts['sleep'], day_index=day_index)
    # Drop Nan Values
    return all_df


# function that does feature extraction for a patient
def extract_user_features(track: Optional[int] = None,
                          patient: Optional[int] = None,
                          mode: Optional[str] = None,
                          num: Optional[int] = None,
                          dtypes: Optional[List] = None,
                          days_flag: str = "intersection",
                          output_format: str = "parquet"):

    if 'all' in dtypes:
        dtypes = ['gyr', 'hrm', 'linacc', 'sleep', 'step']

    # Get days based on the intersection or union of hrm, linacc, gyr.
    days = get_unique_days(track=track, patient=patient, mode=mode, num=num, days_flag=days_flag)

    # Parse all dataframes
    full_dfs = {dtype: parse_data(track, patient, mode, num, dtype) for dtype in dtypes}

    # Create directory
    path_to_save = get_path(track, patient, mode, num) + "/features"
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    day = 0
    p_bar = tqdm(
        days,
        desc=f'Extracting features for each day (Track: {track} | Patient: {patient} | Mode: {mode} | Num: {num})',
        leave=False,
        postfix={"Day": f"{day} | {days[-1]}"})

    for day in p_bar:
        day_features = extract_day_features(full_dfs, day_index=day)
        if output_format == 'parquet':
            out_file = path_to_save + f"/day_{day:02}.parquet"
            day_features.to_parquet(out_file, engine='fastparquet')
        elif output_format == 'csv':
            out_file = path_to_save + f"/day_{day:02}.csv"
            day_features.to_csv(out_file)
        p_bar.set_postfix({"Day": f"{day} | {days[-1]}"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--track', type=int, required=True, choices=[1, 2], help='track 1 or 2')
    parser.add_argument('--patient', type=int, required=True, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9], help='which patient')
    parser.add_argument('--mode', type=str, required=True, choices=["train", "test", "val"], help='which split')
    parser.add_argument('--num', type=int, required=True, help='which number of split')
    parser.add_argument('--dtype', type=str, nargs='+', default=['all'])
    parser.add_argument('--days_flag',
                        default='intersection',
                        choices=['intersection', 'union'],
                        help='Flag that determines which days to keep.')
    parser.add_argument('--output_format',
                        default='parquet',
                        choices=['parquet', 'csv'],
                        help='The output format of the features.')

    args = parser.parse_args()

    extract_user_features(args.track, args.patient, args.mode, args.num, args.dtype, args.days_flag, args.output_format)
