import os
import sys
import argparse

PROJECT_PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, PROJECT_PATH)
import json

import torch

from training.torch_utils import combine_patient_datasets
from models.cnn import SimpleCNN
from training.loops import contrastive_training_loop


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", required=True, help="Json training configuration file.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config_file, "r") as f:
        json_config = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dset, val_dset = combine_patient_datasets(track_id=json_config['track_id'],
                                                    patients=json_config['patients'],
                                                    file_format=json_config['file_format'],
                                                    train_ratio=json_config['split_ratio'],
                                                    window_size=json_config['window_size'],
                                                    feature_mapping=json_config['feature_mapping'],
                                                    upsampling_size=json_config['upsampling_size'])

    model = SimpleCNN(channel_sequence=json_config['channel_sequence'])
    pt_file = os.path.join(PROJECT_PATH, "pretrained_models", json_config["pt_file"])

    contrastive_training_loop(model=model,
                              train_dset=train_dset,
                              val_dset=val_dset,
                              epochs=json_config['epochs'],
                              batch_size=json_config['batch_size'],
                              learning_rate=json_config['learning_rate'],
                              patience=json_config['patience'],
                              pt_file=pt_file,
                              device=device,
                              num_workers=json_config['num_workers'])
