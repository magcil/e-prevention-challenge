import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from tqdm import tqdm

from preprocess.baseline_feature_extraction import extract_user_features
from utils import parse as parser


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--track', type=int, choices=[1, 2], help='The track number.')
    parser.add_argument('--dtypes',
                        nargs='+',
                        default=['hrm', 'gyr', 'linacc'],
                        help='List of data types to be used for extraction.')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    track_id = args.track
    dtypes = args.dtypes

    if track_id == 1:
        patients = list(range(1, 10))
    elif track_id == 2:
        patients = list(range(1, 8))

    p_bar = tqdm(patients, desc='Extracting features for each patient.', total=len(patients))

    for patient_id in p_bar:
        patient_path = parser.get_path(track_id, patient_id)
        patient_dirs = list(map(lambda x: (x[0], int(x[1])), [y.split("_") for y in next(os.walk(patient_path))[1]]))

        for mode, num in patient_dirs:
            extract_user_features(track_id, patient_id, mode, num, dtypes)
