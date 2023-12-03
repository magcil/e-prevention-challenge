# Convolutional Autoencoder
import torch
import torch.nn as nn
import numpy as np

# Define the autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(8, 16, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 
                               kernel_size=(3, 3), 
                               stride=2, 
                               padding=0,
                               output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 
                               kernel_size=(3, 3), 
                               stride=2, 
                               padding=(1,0),
                               output_padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 
                               kernel_size=(3, 3), 
                               stride=2, 
                               padding=(1, 0),
                               output_padding=(1,0)),    
            nn.BatchNorm2d(1),
        )

         
    def forward(self, x, flag=False, save_path=''):
        x = self.encoder(x)
        if flag:
            with open(save_path + '.npy', 'wb') as f:
                np.save(f, x.cpu().detach().numpy())
        x = self.decoder(x)
        return x