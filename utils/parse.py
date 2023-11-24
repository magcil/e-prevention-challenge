import os
from typing import Dict, Optional, List
import glob

import numpy as np
import pandas as pd

from utils import DATA_PATH, DATA_FILES


def get_path(track: Optional[int] = None,
             patient: Optional[int] = None,
             mode: Optional[str] = None,
             num: Optional[int] = None,
             dtype: Optional[str] = None) -> str:
    """Generates specified path incrementally
    
    Args:
        track (int): the track number.
        patient (int): the number of patient.
        mode (str): train/test/val.
        num (int): mode's number.
        dtype (str): the data type (gyr, hrm, etc).

    Returns:
        str: The path of the target file.
    """
    if dtype != None:
        return os.path.join(DATA_PATH, "track_0" + str(track), "P" + str(patient), mode + "_" + str(num),
                            dtype + ".parquet")
    elif num != None:
        return os.path.join(DATA_PATH, "track_0" + str(track), "P" + str(patient), mode + "_" + str(num))
    elif mode != None:
        return os.path.join(DATA_PATH, "track_0" + str(track), "P" + str(patient), mode + "_")
    elif patient != None:
        return os.path.join(DATA_PATH, "track_0" + str(track), "P" + str(patient))
    elif track != None:
        return os.path.join(DATA_PATH, "track_0" + str(track))
    else:
        return DATA_PATH


def parse_data(track: int, patient: int, mode: str, num: int, dtype: str) -> pd.DataFrame:
    """Reads parquet file as dataframe
    
    Args:
        track (int): the track number.
        patient (int): the number of patient.
        mode (str): train/test/val.
        num (int): mode's number.
        dtype (str): the data type (gyr, hrm, etc).

    Returns:
        pd.Dataframe
    """

    return pd.read_parquet(get_path(track, patient, mode, num, dtype), engine="fastparquet")


def parse_mode_data(track: int, patient: int, mode: str, num: int) -> Dict[str, pd.DataFrame]:
    """Reads all parquet files of a model (eg. track_01/P1/train_0/*.parquet)
    
    Args:
        track (int): the track number.
        patient (int): the number of patient.
        mode (str): train/test/val.
        num (int): mode's number.
    
    Returns:
        Dict[str, pd.Dataframe]: Keys are type of data, values dataframes
    """
    d = {}
    for file in DATA_FILES:
        d[file] = parse_data(track, patient, mode, num, file)
    return d


def get_relapses(track: int, patient: int, num: int) -> pd.DataFrame:
    """Return the relapse data of patient's (only val)
    
    Args:
        track (int): the track number.
        patient (int): the number of patient.
        number (int): val set number.
    
    Returns:
        pd.Dataframe
    """
    file_path = os.path.join(get_path(track, patient, "val", num), "relapses.csv")

    return pd.read_csv(file_path)


def iter_on_patient_data(track: int, patient: int, mode: str, dtype: str):
    """Creates an iterator over the data for each track/mode/
    
    Args
        track (int): the track number.
        patient (int): patient id.
        mode (str): train/test/val.
    
    Returns:
        iterator: pd.DataFrame
    """
    modes = glob.glob(os.path.join(get_path(track, patient), mode + "*"))
    for m in modes:
        num = int(m[-1])
        yield pd.read_parquet(get_path(track, patient, m, num, dtype))


def get_unique_days(track: int, patient: int, mode: str, num: int):
    df_1 = parse_data(track, patient, mode, num, "hrm")
    df_2 = parse_data(track, patient, mode, num, "gyr")

    return np.intersect1d(df_1['day_index'].unique(), df_2['day_index'].unique(), assume_unique=True)


def parse_dtypes(track: int, patient: int, mode: str, num: int, dtypes: List[str]) -> Dict[str, pd.DataFrame]:

    return {dtype: parse_data(track, patient, mode, num, dtype) for dtype in dtypes}


def get_features(track_id: int, patient_id: Optional[int] = None, mode: Optional[str] = None, extension=".parquet"):
    """Get all features for the specified path"""
    path = get_path(track=track_id, patient=patient_id)
    target_dirs = [dir[0] for dir in os.walk(path) if os.path.basename(dir[0]).startswith(mode)]
    tree = []

    for dir in target_dirs:
        files = next(os.walk(dir + "/features"))[2]
        for _file in files:
            if _file.endswith(extension) and _file.startswith("day"):
                tree.append(os.path.join(dir, _file))
    return tree
