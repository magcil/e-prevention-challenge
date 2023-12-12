import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import argparse

from tqdm import tqdm
import pandas as pd

import utils.parse as parser


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--track_id', type=int, default=1, help='Track id.')
    parser.add_argument('--patients',
                        nargs='+',
                        default=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                        type=int,
                        help='The patients to check.')
    parser.add_argument('--dtypes',
                        nargs='+',
                        default=['hrm', 'gyr', 'linacc'],
                        help='Data types to take into account.')

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    track_id = args.track_id
    patients = args.patients

    for patient in tqdm(patients, desc='Processing each patient'):
        patient_dirs = parser.get_patient_dirs(track_id, patient)
        for dir in patient_dirs:
            booleans = {dtype: [] for dtype in args.dtypes}
            booleans['day_index'] = []
            mode = dir.split("/")[-1].split("_")
            num, mode = int(mode[1]), mode[0]

            relapse_df = parser.get_relapses(track_id, patient, num, mode)
            # Drop last day - invalid entry
            relapse_df = relapse_df[:-1]
            df_dicts = parser.parse_dtypes(track_id, patient, mode, num, args.dtypes)

            for day_index in relapse_df['day_index']:
                booleans['day_index'].append(day_index)
                for dtype in args.dtypes:
                    df = df_dicts[dtype]
                    if df[df['day_index'] == day_index].empty:
                        booleans[dtype].append(0)
                    else:
                        booleans[dtype].append(1)

            df_to_write = pd.DataFrame(data=booleans)
            df_to_write.to_csv(os.path.join(parser.get_path(track_id, patient, mode, num), "unique_days.csv"))
