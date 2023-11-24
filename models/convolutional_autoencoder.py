# Convolutional Autoencoder

import torch
import torch.nn as nn
import numpy as np

# Define the autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self, ws):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(8, 16, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 
                               kernel_size=(3, 3), 
                               stride=2, 
                               padding=0,
                               output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 
                               kernel_size=(3, 3), 
                               stride=2, 
                               padding=(1,0),
                               output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 
                               kernel_size=(3, 3), 
                               stride=2, 
                               padding=(1,0),
                               output_padding=(1,0)),               
            nn.Sigmoid()
        )
         
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x