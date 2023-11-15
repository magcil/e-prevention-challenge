import os
import datetime
import pandas as pd
import numpy as np
import pyhrv
import scipy
import argparse
from typing import Dict, Optional
import sys

sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.realpath(__file__)), '../'))
from utils.parse import parse_data, get_path

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
    return np.sqrt(df['X'] ** 2 + df['Y'] ** 2 + df['Z'] ** 2).mean()


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

    df_hrm = df_hrm.groupby([df_hrm['day_index'], pd.Grouper(key='DateTime', freq='5Min')]).agg(
        {'heartRate': np.nanmean, 'rRInterval': [np.nanmean, rmssd, sdnn, lombscargle_power_high]})
    return df_hrm

def extract_linacc(df_linacc):
    # where acc is out of limits, set it to nan
    df_linacc.loc[
        (df_linacc['X'] < valid_ranges['acc_X'][0]) | (df_linacc['X'] >= valid_ranges['acc_X'][1]), 'X'] = np.nan
    df_linacc.loc[
        (df_linacc['Y'] < valid_ranges['acc_Y'][0]) | (df_linacc['Y'] >= valid_ranges['acc_Y'][1]), 'Y'] = np.nan
    df_linacc.loc[
        (df_linacc['Z'] < valid_ranges['acc_Z'][0]) | (df_linacc['Z'] >= valid_ranges['acc_Z'][1]), 'Z'] = np.nan

    df_linacc = df_linacc.groupby([df_linacc['day_index'], pd.Grouper(key='DateTime', freq='5Min')]).apply(get_norm)

    return df_linacc

def extract_sleep(df):
    return df
def extract_gyr(df):
    return df
def extract_step(df):
    return df

FEATURE_FUNC = {
    'gyr': extract_gyr,
    'hrm': extract_hr,
    'linacc': extract_linacc,
    'sleep': extract_sleep,
    'step': extract_step
}

# function that does feature extraction for a patient
def extract_user_features(track: Optional[int] = None,
                          patient: Optional[int] = None,
                          mode: Optional[str] = None,
                          num: Optional[int] = None,
                          dtypes: Optional[list] = None):


    if 'all' in dtypes:
       dtypes = ['gyr', 'hrm', 'linacc', 'sleep', 'step']

    all_df = []
    for dtype in dtypes:
        print(
            'Extracting {} related features for track {}, patient {} and mode {}_{}'.format(dtype, track, patient, mode,
                                                                                            num))
        df = parse_data(track, patient, mode, num, dtype)

        # Convert Timedelta to datetime.time
        start_time = datetime.datetime.strptime('00:00:00', '%H:%M:%S').time()  # Start time
        df['DateTime'] = df['time'].apply(
            lambda t: datetime.datetime.combine(datetime.datetime.today(), start_time) + t)
        feature_extractor = FEATURE_FUNC[dtype]
        result_df = feature_extractor(df)
        all_df.append(result_df)

    # combine all
    df = pd.concat(all_df, axis=1)
    df = df.reset_index()

    # create positional encoding features
    h = df['DateTime'].dt.hour
    m = df['DateTime'].dt.minute
    time_value = h * 60 + m
    df['sin_t'] = np.sin(time_value * (2. * np.pi / (60 * 24)))
    df['cos_t'] = np.cos(time_value * (2. * np.pi / (60 * 24)))


    path_to_save = get_path(track, patient, mode, num) + "/features"
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    # Get unique values from the 'day_index' column
    unique_days = df['day_index'].unique()

    # Split DataFrame based on unique values in 'day_index' column and save to .npz files
    for day in unique_days:
        subset_df = df[df['day_index'] == day]  # Subset DataFrame for each unique day
        # Convert subset DataFrame to a NumPy array
        subset_array = subset_df.drop(columns=['DateTime', 'day_index']).to_numpy()
        out_file = path_to_save + f"/{day}.npz"
        # Save the subset array to a .npz file with the category name
        np.savez(out_file, data=subset_array)

    print('Saved features for {} features for track {}, patient {} and mode {}_{}'.format(dtypes, track, patient, mode,
                                                                                            num))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--track', type=int, required=True, choices=[1, 2], help='track 1 or 2')
    parser.add_argument('--patient', type=int, required=True, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9], help='which patient')
    parser.add_argument('--mode', type=str, required=True, choices=["train", "test", "val"], help='which split')
    parser.add_argument('--num', type=int, required=True, help='which number of split')
    parser.add_argument('--dtype', type=str, nargs='+', default=['all'])

    args = parser.parse_args()

    extract_user_features(args.track, args.patient, args.mode, args.num, args.dtype)

