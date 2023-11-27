import os
import sys
sys.path.insert(1, os.path.join(os.path.dirname(sys.path[0])))

import torch
import numpy as np
from tqdm import tqdm
from torchmetrics.regression import MeanSquaredLogError
import sklearn.metrics
from models.convolutional_autoencoder import Autoencoder
from utils.dataset import RelapseDetectionDataset
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
import argparse
import seaborn as sns


class RelapseDetection():

    def __init__(self, train_feats_path, test_feats_path, checkpoint_path, batch_size, patience, lr, epochs, window_size):
        self.train_features_path = train_feats_path
        self.test_features_path = test_feats_path
        self.BEST_MODEL_PATH = checkpoint_path
        self.batch_size = batch_size
        self.early_stopping_patience = patience
        self.lr = float(lr)
        self.epochs = epochs
        self.window_size = int(window_size)

        # define the model --> Convolutional Autoencoder
        self.model = Autoencoder(self.window_size)

    def select_device(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        device = 'cpu'
        return device
    
    # add a collate fn which ignores None
    def collate_fn(self, batch):
        batch = [x for x in batch if x is not None]
        if len(batch) == 0:
            return None
        return torch.utils.data.dataloader.default_collate(batch)
    
    def train(self, model, criterion, optimizer, loader, device, epoch):

        self.model.train() # switch into training mode
        running_loss = 0 # define loss
        batch_counter = 0 # define batch counter

        # and start the training loop!
        
        for _, data in enumerate(loader, 0):
            feature_vector = data[0]

            batch_counter += 1

            feature_vector = feature_vector.to(device)

            optimizer.zero_grad()
            output = self.model(feature_vector)

            loss = criterion(output, feature_vector)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        epoch_loss = running_loss / batch_counter
        self.scheduler.step()

        if (epoch%100 == 0):
            print(f'Learning rate at epoch {epoch}: {optimizer.param_groups[0]["lr"]}')

        print('Epoch [{}], Training Loss: {:.4f}'.format(epoch, epoch_loss))

    
    def validate(self, model, criterion, loader, device, epoch):

        self.model.eval() # switch into evaluation mode
        running_loss = 0 # define loss
        batch_counter = 0 # define batch counter

        # and start the training loop!
        
        for _, data in enumerate(loader, 0):
            feature_vector = data[0]

            batch_counter += 1

            feature_vector = feature_vector.to(device)

            #optimizer.zero_grad()
            output = self.model(feature_vector)

            loss = criterion(output, feature_vector)

            running_loss += loss.item()
        
        epoch_loss = running_loss / batch_counter

        if (epoch_loss < self.best_loss):
            self.best_loss = epoch_loss
            self.early_stopping_counter = 0
            # save model in order to retrieve at the end...
            torch.save(model.state_dict(), self.BEST_MODEL_PATH)
        else:
            self.early_stopping_counter += 1

        print('Epoch [{}], Validation Loss: {:.4f}'.format(epoch, epoch_loss))


    def test(self, model, loader, device):
        model.eval()
        originals, reconstructions = list(), list()
        batch_counter = 0
        for _, data in enumerate(loader, 0):
            feature_vector = data[0]
            batch_counter += 1

            feature_vector = feature_vector.to(device)

            reconstruction = model(feature_vector)
            
            originals.append(feature_vector)
            reconstructions.append(reconstruction)

        return originals, reconstructions

    def run(self):

        # Load dataset
        train_dataset = RelapseDetectionDataset(self.train_features_path, self.window_size, split='train')
        val_dataset = RelapseDetectionDataset(self.train_features_path, self.window_size, split='dev')
        test_dataset = RelapseDetectionDataset(self.test_features_path, self.window_size, split='validation')

        # Define the dataloader
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                                batch_size=self.batch_size, 
                                                shuffle=True,
                                                collate_fn=self.collate_fn)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                                    batch_size=self.batch_size, 
                                                    shuffle=False,
                                                    collate_fn=self.collate_fn)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                                batch_size=self.batch_size,
                                                collate_fn=self.collate_fn)
        
        self.device = self.select_device()
        self.model.to(self.device)

        # Define the loss function and optimizer
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.996)

        self.early_stopping_counter = 0
        self.best_loss = 100000 # just a random large value

        for epoch in range(self.epochs):
            self.train(self.model, criterion, optimizer, train_loader, self.device, epoch)
            self.validate(self.model, criterion, val_loader, self.device, epoch)
            print('early stopping counter:', self.early_stopping_counter)
            if ((self.early_stopping_counter >= self.early_stopping_patience) or (epoch == (self.epochs - 1))):
                print(f'Training ended. Best MSE loss on validation data {self.best_loss}')
                best_model = Autoencoder(self.window_size)
                state_dict = torch.load(self.BEST_MODEL_PATH)
                best_model.load_state_dict(state_dict)
                best_model.to(self.device)
                break

        print('Now predicting on unseen validation set...')

        originals, reconstructions = self.test(best_model, test_loader, self.device)
        
        anomaly_scores = [criterion(originals[i], reconstructions[i]).item() for i in range(len(originals))]

        print('Min MSE loss on validation data (normal state):', np.min(anomaly_scores))
        print('Average MSE loss on validation data (normal state):', np.mean(anomaly_scores))
        print('Median MSE loss on validation data (normal state):', np.median(anomaly_scores))
        print('Max MSE loss on validation data (normal state):', np.max(anomaly_scores))
      


if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-tp",
        "--train_features_path",
        required=True,
        nargs='+',
        help="path to folder where training .parquet files are contained (one for each day)"
    )

    parser.add_argument(
        "-tep",
        "--test_features_path",
        required=True,
        nargs='+',
        help="path to folder where testing .parquet files are contained (one for each day)",
    )

    parser.add_argument(
        "-c",
        "--checkpoint_path",
        required=True,
        help="path to model load/save checkpoint.",
    )

    parser.add_argument(
        "-bs",
        "--batch_size",
        required=False,
        default=8,
        type=int,
        help="batch size for training, validation and test processes",
    )

    parser.add_argument(
        "-es",
        "--early_stopping",
        required=False,
        default=150,
        type=int,
        help="early stopping patience to use during training",
    )

    parser.add_argument(
        "-lr",
        "--learning_rate",
        required=False,
        default=1e-4,
        type=float,
        help="learning rate to use during training",
    )

    parser.add_argument(
        "-ep",
        "--epochs",
        required=False,
        default=1500,
        type=int,
        help="number of epochs to train for",
    )

    parser.add_argument(
        "-ws",
        "--window_size",
        required=False,
        default=50,
        type=int,
        help="number of 5 minute intervals to use during training",
    )

    args = parser.parse_args()

    obj = RelapseDetection(args.train_features_path,
                           args.test_features_path,
                           args.checkpoint_path,
                           args.batch_size,
                           args.early_stopping,
                           args.learning_rate,
                           args.epochs,
                           args.window_size)
    
    obj.run()