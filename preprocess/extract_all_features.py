import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm

from preprocess.baseline_feature_extraction import extract_user_features
from utils import parse as parser


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--track', type=int, choices=[1, 2], default=1, help='The track number.')
    parser.add_argument('--patients',
                        nargs='+',
                        type=int,
                        default=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                        help='Patients to extract features')
    parser.add_argument('--dtypes',
                        nargs='+',
                        default=['hrm', 'gyr', 'linacc'],
                        help='List of data types to be used for extraction.')
    parser.add_argument('--days_flag',
                        default='intersection',
                        choices=['intersection', 'union'],
                        help="Determine unique days.")
    parser.add_argument('--output_format',
                        default="parquet",
                        choices=['parquet', 'csv'],
                        help='The format to save the features.')
    parser.add_argument("--exclude_test", action="store_true", help="Whether to exclude features for test or not.")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    track_id = args.track
    dtypes = args.dtypes

    patients = args.patients

    p_bar = tqdm(patients, desc='Extracting features for each patient.', total=len(patients))

    for patient_id in p_bar:
        patient_path = parser.get_path(track_id, patient_id)
        patient_dirs = list(map(lambda x: (x[0], int(x[1])), [y.split("_") for y in next(os.walk(patient_path))[1]]))

        # Exclude test features if specified
        if args.exclude_test:
            patient_dirs = [dir for dir in patient_dirs if "test" not in dir[0]]

        for mode, num in patient_dirs:
            extract_user_features(track_id,
                                  patient_id,
                                  mode,
                                  num,
                                  dtypes,
                                  days_flag=args.days_flag,
                                  output_format=args.output_format)
