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
from utils.plots import density_plot, histogram_with_kde
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
import argparse
import seaborn as sns


class RelapseDetection():

    def __init__(self, train_feats_path, test_feats_path, checkpoint_path, batch_size, window_size, save_enc):
        self.train_features_path = train_feats_path
        self.test_features_path = test_feats_path
        self.checkpoint_path = checkpoint_path
        self.batch_size = batch_size
        self.window_size = int(window_size)
        self.save_enc = save_enc

        # define the model --> Convolutional Autoencoder
        self.model = Autoencoder(self.window_size)

    def select_device(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        device='cpu'
        return device
    
    # add a collate fn which ignores None
    def collate_fn(self, batch):
        batch = [x for x in batch if x is not None]
        if len(batch) == 0:
            return None
        return torch.utils.data.dataloader.default_collate(batch)

    def test(self, model, flag, loader, device):
        model.eval()
        originals, reconstructions = list(), list()
        batch_counter = 0

        for _, data in enumerate(loader, 0):
            feature_vector = data[0]
            if flag==True:
                location = os.path.dirname(os.path.dirname(data[1][0])) + '/encodings/' + os.path.dirname(data[1][0]).split('/')[-1] + '/'
                if os.path.exists(location) == False:
                    os.makedirs(location)
                filename = location + data[1][0].split('/')[-1]
            else:
                location = ''
            batch_counter += 1

            feature_vector = feature_vector.to(device)

            reconstruction = model(feature_vector, flag, filename)
            
            originals.append(feature_vector)
            reconstructions.append(reconstruction)

        return originals, reconstructions

    def run(self):

        # Load dataset
        test_dataset = RelapseDetectionDataset(self.test_features_path[0], self.window_size, split='test')
        test_dataset_2 = RelapseDetectionDataset(self.test_features_path[1], self.window_size, split='test')
        train_dataset = RelapseDetectionDataset(self.train_features_path, self.window_size, split='train')
        val_dataset = RelapseDetectionDataset(self.train_features_path, self.window_size, split='validation')

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

        test_loader_2 = torch.utils.data.DataLoader(dataset=test_dataset_2, 
                                                batch_size=1,
                                                collate_fn=self.collate_fn)
        
        self.device = self.select_device()

        # Define the loss function and optimizer
        criterion = torch.nn.MSELoss()

        best_model = Autoencoder(self.window_size)
        state_dict = torch.load(self.checkpoint_path)
        best_model.load_state_dict(state_dict)
        best_model.to(self.device)

        print('Predict on train data')
        originals_train, reconstructions_train = self.test(best_model, self.save_enc, train_loader, self.device)
        anomalies_mse = list()
        
        anomaly_scores_train = [criterion(originals_train[i], reconstructions_train[i]).item() for i in range(len(originals_train))]
        anomalies_mse.append(anomaly_scores_train)

        print('Min MSE loss on train data:', round(np.min(anomaly_scores_train), 4))
        print('Average MSE loss on train data:', round(np.mean(anomaly_scores_train), 4))
        print('Median MSE loss on train data:', round(np.median(anomaly_scores_train), 4))
        print('Max MSE loss on train data:', round(np.max(anomaly_scores_train), 4))

        print('Now predicting on test set...')

        originals_val_normal, reconstructions_val_normal = self.test(best_model, self.save_enc, test_loader, self.device)
        
        anomaly_scores_val_normal = [criterion(originals_val_normal[i], reconstructions_val_normal[i]).item() for i in range(len(originals_val_normal))]
        anomalies_mse.append(anomaly_scores_val_normal)

        print('Min MSE loss on test data (normal state):', round(np.min(anomaly_scores_val_normal), 4))
        print('Average MSE loss on test data (normal state):', round(np.mean(anomaly_scores_val_normal), 4))
        print('Median MSE loss on test data (normal state):', round(np.median(anomaly_scores_val_normal), 4))
        print('Max MSE loss on test data (normal state):', round(np.max(anomaly_scores_val_normal),4))

        originals_val_relapsed, reconstructions_val_relapsed = self.test(best_model, self.save_enc, test_loader_2, self.device)
        
        anomaly_scores_val_relapsed = [criterion(originals_val_relapsed[i], reconstructions_val_relapsed[i]).item() for i in range(len(originals_val_relapsed))]
        anomalies_mse.append(anomaly_scores_val_relapsed)
        print('Min MSE loss on test data (relapsed state):', round(np.min(anomaly_scores_val_relapsed), 4))
        print('Average MSE loss on test data (relapsed state):', round(np.mean(anomaly_scores_val_relapsed), 4))
        print('Median MSE loss on test data (relapsed state):', round(np.median(anomaly_scores_val_relapsed), 4))
        print('Max MSE loss on test data (relapsed state):', round(np.max(anomaly_scores_val_relapsed), 4))
        

        density_plot(to_plot=anomalies_mse,
                    labels=["MSE train","MSE val_0 (normal)", "MSE val_1 (relapsed)"],
                    colors=["dodgerblue", "deeppink", "gold"],
                    save_path= os.getcwd() + '/figs')


        histogram_with_kde(to_plot=anomalies_mse, bins=10,
                            labels=["MSE_train","MSE_val_0", "MSE_val_1"],
                            colors=["dodgerblue", "deeppink", "gold"],
                            save_path= os.getcwd() +  '/figs')





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
        help="path to model saved checkpoint.",
    )

    parser.add_argument(
        "-bs",
        "--batch_size",
        required=False,
        default=1,
        type=int,
        help="batch size for training, validation and test processes",
    )


    parser.add_argument(
        "-ws",
        "--window_size",
        required=False,
        default=50,
        type=int,
        help="number of 5 minute intervals to use during training",
    )

    parser.add_argument(
        "-enc",
        "--save_encodings",
        required=False,
        default=False,
        action="store_true",
        help="whether to store or not the encodings of the CAE",
    )

    args = parser.parse_args()


    obj = RelapseDetection(args.train_features_path,
                           args.test_features_path,
                           args.checkpoint_path,
                           args.batch_size,
                           args.window_size,
                           args.save_encodings)
    
    obj.run()