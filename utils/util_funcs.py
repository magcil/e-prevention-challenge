import os
from typing import List, Tuple
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from utils import parse



def get_pos_neg_samples(track_id: int, patient_id: int, feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Get relapse feature and non relapse feature for a given track/patient
    
    Args:
        track_id (int): Track id
        patient_id (int): Patient id
        feature_names (List[str]): The feature names

    Returns:
        Tuple[nd.array, nd.array]: relapse_features, non_relapse_features
    """
    trg_path = parse.get_path(track_id, patient_id)
    vals = [dir.split('_') for dir in os.listdir(trg_path) if dir.startswith("val")]
    all_pos, all_neg = [], []
    for _, num in vals:
        relapse_csv = parse.get_relapses(track_id, patient_id, int(num))
        items = parse.get_features(track_id, patient_id, "val", int(num))
        relapse_days = relapse_csv[relapse_csv['relapse'] == 1]['day_index'].to_list()

        for item in items:
            day_index = int(os.path.basename(item).split("_")[1][:2])

            if day_index in relapse_days:
                df = pd.read_parquet(item, engine='fastparquet')
                df.sort_values(by='DateTime')
                all_pos.append(df[feature_names].to_numpy())
            else:
                df = pd.read_parquet(item, engine='fastparquet')
                df.sort_values(by='DateTime')
                x = df[feature_names].to_numpy()
                all_neg.append(x)

    all_pos, all_neg = np.concatenate(all_pos, axis=0), np.concatenate(all_neg, axis=0)

    return all_pos[~np.isinf(all_pos).any(1)], all_neg[~np.isinf(all_neg).any(1)]