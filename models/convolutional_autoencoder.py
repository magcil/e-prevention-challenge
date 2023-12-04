# Convolutional Autoencoder
import torch
import torch.nn as nn
import numpy as np


# Define the autoencoder architecture
class Autoencoder(nn.Module):

    def __init__(self):
        super(Autoencoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 8, kernel_size=(3, 3), stride=1, padding=1), nn.BatchNorm2d(8),
                                   nn.ReLU())
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.conv2 = nn.Sequential(nn.Conv2d(8, 16, kernel_size=(3, 3), stride=1, padding=1), nn.BatchNorm2d(16),
                                   nn.ReLU())
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.conv3 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=(3, 3), stride=1, padding=1), nn.BatchNorm2d(32),
                                   nn.ReLU())
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.unconv1 = nn.Sequential(nn.ConvTranspose2d(32, 16, kernel_size=(3, 3), stride=1, padding=1),
                                     nn.BatchNorm2d(16), nn.ReLU())
        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.unconv2 = nn.Sequential(nn.ConvTranspose2d(16, 8, kernel_size=(3, 3), stride=1, padding=1),
                                     nn.BatchNorm2d(8), nn.ReLU())
        self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.unconv3 = nn.ConvTranspose2d(8, 1, kernel_size=(3, 3), stride=1, padding=1)

    def forward(self, x, flag=False, save_path=''):
        indices = []
        x = self.conv1(x)
        x, index = self.pool1(x)
        indices.append(index)
        x = self.conv2(x)
        x, index = self.pool2(x)
        indices.append(index)
        x = self.conv3(x)
        x, index = self.pool3(x)
        indices.append(index)

        if flag:
            with open(save_path + '.npy', 'wb') as f:
                np.save(f, x.cpu().detach().numpy())

        x = self.unpool1(x, indices[2])
        x = self.unconv1(x)
        x = self.unpool2(x, indices[1])
        x = self.unconv2(x)
        x = self.unpool3(x, indices[0])
        x = self.unconv3(x)

        return x
