import os
from typing import List, Tuple, Optional
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc

from utils import parse


def get_pos_neg_samples(track_id: int,
                        patient_id: int,
                        feature_names: List[str],
                        group_labels: Optional[bool] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Get relapse feature and non relapse feature for a given track/patient
    
    Args:
        track_id (int): Track id
        patient_id (int): Patient id
        feature_names (List[str]): The feature names
        group_labels (List[str]): Whether to return unique labels of val + day for each 5Min sample

    Returns:
        Dict[str, np.ndarray]: Keys give information for each value np.ndarray
    """
    trg_path = parse.get_path(track_id, patient_id)
    vals = [dir.split('_') for dir in os.listdir(trg_path) if dir.startswith("val")]
    all_pos, all_neg = [], []

    if group_labels:
        groups = []

    for _, num in vals:
        relapse_csv = parse.get_relapses(track_id, patient_id, int(num))
        items = parse.get_features(track_id, patient_id, "val", int(num))
        relapse_days = relapse_csv[relapse_csv['relapse'] == 1]['day_index'].to_list()

        for item in items:
            day_index = int(os.path.basename(item).split("_")[1][:2])

            if day_index in relapse_days:
                df = pd.read_parquet(item, engine='fastparquet')
                df.replace(to_replace=[-np.inf, np.inf], value=np.nan, inplace=True)
                df.dropna(inplace=True)
                df.sort_values(by='DateTime')
                x = df[feature_names].to_numpy()
                x = x[~np.isinf(x).any(1)]
                all_pos.append(x)
                if group_labels:
                    groups.append(x.shape[0] * [f"P{patient_id}/val_" + num + f"/day_{day_index:02}"])
            else:
                df = pd.read_parquet(item, engine='fastparquet')
                df.replace(to_replace=[-np.inf, np.inf], value=np.nan, inplace=True)
                df.dropna(inplace=True)
                df.sort_values(by='DateTime')
                x = df[feature_names].to_numpy()
                x = x[~np.isinf(x).any(1)]
                all_neg.append(x)
                if group_labels:
                    groups.append(x.shape[0] * [f"P{patient_id}/val_" + num + f"/day_{day_index:02}"])

    if group_labels:
        return {
            "relapses": np.concatenate(all_pos, axis=0),
            "non_relapses": np.concatenate(all_neg, axis=0),
            "groups": np.concatenate(groups, axis=0)
        }
    else:
        return {"relapses": np.concatenate(all_pos, axis=0), "non_relapses": np.concatenate(all_neg, axis=0)}


def fill_predictions(track_id, patient_id, anomaly_scores, split, days):
    mode, num = split.split("_")[0], int(split.split("_")[1])
    df_relapses = parse.get_relapses(track=track_id, patient=patient_id, num=num, mode=mode)
    # Drop last row from relapses
    df_relapses = df_relapses.iloc[:-1]

    df_scores = pd.DataFrame({"anomaly_scores": anomaly_scores, "day_index": days})
    df_scores.sort_values(by="day_index", inplace=True)

    # Merge
    final_df = df_relapses.merge(df_scores, how="outer", on="day_index")
    final_df['anomaly_scores'] = final_df['anomaly_scores'].interpolate()

    # Interpolate to fill na values
    return final_df.ffill().bfill()


def calculate_roc_pr_auc(anomaly_scores, labels):
    # Compute metrics
    precision, recall, _ = precision_recall_curve(labels, anomaly_scores)

    fpr, tpr, _ = roc_curve(labels, anomaly_scores)

    return {"ROC AUC": auc(fpr, tpr), "PR AUC": auc(recall, precision)}


def svm_score(preds, dist_from_hp):
    if len(dist_from_hp) > 1:
        min_, max_ = dist_from_hp.min(), dist_from_hp.max()
        median = 1 if max_ == min_ else ((dist_from_hp.median() - min_) / (max_ - min_)) + 1
    else:
        median = 1
    score = median * (preds[preds == -1].shape[0] / len(preds))

    return min(score, 1)
