import os
import sys

sys.path.insert(1, os.path.join(os.path.dirname(sys.path[0])))

import torch

from models.convolutional_autoencoder import Autoencoder
from utils.dataset import RelapseDetectionDataset
from utils.split import split_train_val, handle_dev
import torch.optim.lr_scheduler as lr_scheduler
import argparse
import pickle

import warnings

warnings.filterwarnings("ignore")


class RelapseDetection():

    def __init__(self, train_feats_path, dev_feats_path, checkpoint_path, batch_size, patience, lr, epochs, window_size,
                 samples_per_day, format = 'parquet'):
        self.train_features_path = train_feats_path
        self.dev_features_path = dev_feats_path
        self.BEST_MODEL_PATH = checkpoint_path
        self.batch_size = batch_size
        self.early_stopping_patience = patience
        self.lr = float(lr)
        self.epochs = epochs
        self.window_size = int(window_size)
        self.spd = samples_per_day
        self.format = format

        # define the model --> Convolutional Autoencoder
        self.model = Autoencoder()

    def select_device(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        return device

    # add a collate fn which ignores None
    def collate_fn(self, batch):
        batch = [x for x in batch if x is not None]
        if len(batch) == 0:
            return None
        return torch.utils.data.dataloader.default_collate(batch)

    def train(self, model, criterion, optimizer, loader, device, epoch):

        self.model.train()  # switch into training mode
        running_loss = 0  # define loss
        batch_counter = 0  # define batch counter

        # and start the training loop!

        for _, data in enumerate(loader, 0):
            feature_vector = data[0]

            batch_counter += 1

            feature_vector = feature_vector.to(device)

            optimizer.zero_grad()
            output = self.model(feature_vector)
            #print('out:', output)

            loss = criterion(output, feature_vector)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / batch_counter
        """if optimizer.param_groups[0]["lr"] > 5e-5:
            self.scheduler.step(epoch_loss)"""

        if (epoch % 50 == 0):
            print(f'Learning rate at epoch {epoch}: {optimizer.param_groups[0]["lr"]}')

        print('Epoch [{}], Training Loss: {:.4f}'.format(epoch, epoch_loss))

    def validate(self, model, criterion, loader, device, epoch, optimizer):

        self.model.eval()  # switch into evaluation mode
        running_loss = 0  # define loss
        batch_counter = 0  # define batch counter

        # and start the training loop!
        with torch.no_grad():
            for _, data in enumerate(loader, 0):
                feature_vector = data[0]

                batch_counter += 1

                feature_vector = feature_vector.to(device)

                #optimizer.zero_grad()
                output = self.model(feature_vector)

                loss = criterion(output, feature_vector)

                running_loss += loss.item()

        epoch_loss = running_loss / batch_counter

        if optimizer.param_groups[0]["lr"] > 5e-5:
            self.scheduler.step(epoch_loss)

        if (epoch_loss < self.best_loss - 1e-6):
            self.best_loss = epoch_loss
            self.early_stopping_counter = 0
            # save model in order to retrieve at the end...
            torch.save(model.state_dict(), self.BEST_MODEL_PATH)
        else:
            self.early_stopping_counter += 1

        print('Epoch [{}], Validation Loss: {:.4f}'.format(epoch, epoch_loss))
        return epoch_loss

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

    def common_data(self, list1, list2):
        result = False

        # traverse in the 1st list
        for x in list1:

            # traverse in the 2nd list
            for y in list2:

                # if one common
                if x == y:
                    print(x)
                    result = True
                    return result

        return result

    def run(self):

        train_patient_dir = os.path.dirname(os.path.dirname(self.train_features_path[0]))
        dev_patient_dir = os.path.dirname(os.path.dirname(self.dev_features_path[0]))
        train_paths, val_paths = split_train_val(self.train_features_path)
        dev_paths = handle_dev(self.dev_features_path)

        # Load dataset
        _ = dict()
        self.train_dataset = RelapseDetectionDataset(train_paths,
                                                     train_patient_dir,
                                                     self.window_size,
                                                     self.spd,
                                                     _,
                                                     _,
                                                     split='train',
                                                     calc_norm=True, 
                                                     format=self.format)
        self.val_dataset = RelapseDetectionDataset(val_paths,
                                                   train_patient_dir,
                                                   self.window_size,
                                                   self.spd,
                                                   self.train_dataset.means,
                                                   self.train_dataset.stds,
                                                   split='validation',
                                                   format=self.format)

        output = open(self.BEST_MODEL_PATH.split('.')[0] + '-means.pkl', 'wb')
        pickle.dump(self.train_dataset.means, output)
        output.close()

        output = open(self.BEST_MODEL_PATH.split('.')[0] + '-stds.pkl', 'wb')
        pickle.dump(self.train_dataset.stds, output)
        output.close()

        # Define the dataloader
        train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=True,
                                                   collate_fn=self.collate_fn,
                                                   num_workers=8)
        val_loader = torch.utils.data.DataLoader(dataset=self.val_dataset,
                                                 batch_size=self.batch_size,
                                                 shuffle=False,
                                                 collate_fn=self.collate_fn,
                                                 num_workers=8)

        if (not dev_paths[0]) == False:  # not 'list_name' returns True if the list is empty
            self.dev_dataset_normal = RelapseDetectionDataset(dev_paths[0],
                                                              dev_patient_dir,
                                                              self.window_size,
                                                              1,
                                                              self.train_dataset.means,
                                                              self.train_dataset.stds,
                                                              split='dev',
                                                              state='normal',
                                                              format=self.format)
            dev_loader_normal = torch.utils.data.DataLoader(dataset=self.dev_dataset_normal,
                                                            batch_size=self.batch_size,
                                                            collate_fn=self.collate_fn,
                                                            num_workers=8)

        if (not dev_paths[1]) == False:  # not 'list_name' returns True if the list is empty
            self.dev_dataset_relapsed = RelapseDetectionDataset(dev_paths[1],
                                                                dev_patient_dir,
                                                                self.window_size,
                                                                1,
                                                                self.train_dataset.means,
                                                                self.train_dataset.stds,
                                                                split='dev',
                                                                state='relapsed',
                                                                format=self.format)
            dev_loader_relapsed = torch.utils.data.DataLoader(dataset=self.dev_dataset_relapsed,
                                                              batch_size=self.batch_size,
                                                              collate_fn=self.collate_fn,
                                                              num_workers=8)

        self.device = self.select_device()
        self.model.to(self.device)

        # Define the loss function and optimizer
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        #self.scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                        factor=0.1,
                                                        patience=3,
                                                        cooldown=1,
                                                        min_lr=1e-5,
                                                        verbose=True)

        self.early_stopping_counter = 0
        self.best_loss = 100000  # just a random large value

        for epoch in range(self.epochs):
            self.train(self.model, criterion, optimizer, train_loader, self.device, epoch)
            epoch_loss = self.validate(self.model, criterion, val_loader, self.device, epoch, optimizer)
            self.scheduler.step(epoch_loss)

            if ((self.early_stopping_counter >= self.early_stopping_patience) or (epoch == (self.epochs - 1))):
                print(f'Training ended. Best MSE loss on validation data {self.best_loss}')
                best_model = Autoencoder()
                state_dict = torch.load(self.BEST_MODEL_PATH)
                best_model.load_state_dict(state_dict)
                best_model.to(self.device)
                break

        print('Now predicting on unseen validation set...')
        dev_criterion = torch.nn.MSELoss(reduction='none')  # 'none' to keep per-element losses

        if 'dev_loader_normal' in locals():
            originals, reconstructions = self.test(best_model, dev_loader_normal, self.device)
            #anomaly_scores = dev_criterion(originals[0], reconstructions[0])
            # Calculate mean squared error (MSE) loss per element for each sample
            mse_loss_per_sample = dev_criterion(originals[0],
                                                reconstructions[0]).mean(dim=(2, 3))  # Reduction along height and width

            print('Min MSE loss on dev data (normal state):', mse_loss_per_sample.min().item())
            print('Average MSE loss on dev data (normal state):', mse_loss_per_sample.mean().item())
            print('Median MSE loss on dev data (normal state):', mse_loss_per_sample.median().item())
            print('Max MSE loss on dev data (normal state):', mse_loss_per_sample.max().item())

        if 'dev_loader_relapsed' in locals():
            originals, reconstructions = self.test(best_model, dev_loader_relapsed, self.device)

            #anomaly_scores = dev_criterion(originals[0], reconstructions[0])
            # Calculate mean squared error (MSE) loss per element for each sample
            mse_loss_per_sample = dev_criterion(originals[0],
                                                reconstructions[0]).mean(dim=(2, 3))  # Reduction along height and width

            print('Min MSE loss on dev data (relapsed state):', mse_loss_per_sample.min().item())
            print('Average MSE loss on dev data (relapsed state):', mse_loss_per_sample.mean().item())
            print('Median MSE loss on dev data (relapsed state):', mse_loss_per_sample.median().item())
            print('Max MSE loss on dev data (relapsed state):', mse_loss_per_sample.max().item())


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-tp",
                        "--train_features_path",
                        required=True,
                        nargs='+',
                        help="path to folder where training .parquet files are contained (one for each day)")

    parser.add_argument(
        "-tep",
        "--dev_features_path",
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
        default=32,
        type=int,
        help="batch size for training, validation and test processes",
    )

    parser.add_argument(
        "-es",
        "--early_stopping",
        required=False,
        default=20,
        type=int,
        help="early stopping patience to use during training",
    )

    parser.add_argument(
        "-lr",
        "--learning_rate",
        required=False,
        default=1e-3,
        type=float,
        help="learning rate to use during training",
    )

    parser.add_argument(
        "-ep",
        "--epochs",
        required=False,
        default=500,
        type=int,
        help="number of epochs to train for",
    )

    parser.add_argument(
        "-ws",
        "--window_size",
        required=False,
        default=48,
        type=int,
        help="number of 5 minute intervals to use during training",
    )

    parser.add_argument(
        "-spd",
        "--samples_per_day",
        required=False,
        default=5,
        type=int,
        help="number of 5 minute intervals to use during training",
    )

    parser.add_argument('-f',
                        '--file_format',
                        choices=['parquet', 'csv'],
                        default='parquet',
                        help='The file format of features.')

    args = parser.parse_args()

    obj = RelapseDetection(args.train_features_path, args.dev_features_path, args.checkpoint_path, args.batch_size,
                           args.early_stopping, args.learning_rate, args.epochs, args.window_size, args.samples_per_day,
                           format = args.file_format)

    obj.run()
