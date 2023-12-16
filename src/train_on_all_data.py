import os
import sys
import argparse
import json
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset

from training.loops import autoencoder_train_loop
from models.convolutional_autoencoder import Autoencoder
from training.torch_utils import combine_patient_datasets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='Training config file.')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    with open(args.config, "r") as f:
        json_config = json.load(f)
    track_id = json_config["track_id"]
    patients = json_config["patients"]
    feature_mapping = json_config["feature_mapping"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dset, val_dset = combine_patient_datasets(track_id=track_id,
                                                    patients=patients,
                                                    file_format=json_config['file_format'],
                                                    train_ratio=json_config['train_ratio'],
                                                    window_size=json_config['window_size'],
                                                    feature_mapping=feature_mapping,
                                                    upsampling_size=json_config['upsampling_size'])

    # Initialize model
    model = Autoencoder()
    model = model.to(device)

    pt_file = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
                           json_config["pretrained_models"],
                           f"track_{track_id}_on_all_data_{datetime.today().date()}.pt")

    # Start training
    autoencoder_train_loop(train_dset=train_dset,
                           val_dset=val_dset,
                           model=model,
                           epochs=json_config["epochs"],
                           batch_size=json_config["batch_size"],
                           patience=json_config["patience"],
                           learning_rate=json_config["learning_rate"],
                           pt_file=pt_file,
                           device=device,
                           num_workers=json_config["num_workers"])
