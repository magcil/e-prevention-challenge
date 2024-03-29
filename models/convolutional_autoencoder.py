# Convolutional Autoencoder
import torch
import torch.nn as nn
import numpy as np


# Define the autoencoder architecture
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

        emb = x

        if flag:
            with open(save_path + '.npy', 'wb') as f:
                np.save(f, x.cpu().detach().numpy())

        x = self.unpool1(x, indices[2])
        x = self.unconv1(x)
        x = self.unpool2(x, indices[1])
        x = self.unconv2(x)
        x = self.unpool3(x, indices[0])
        x = self.unconv3(x)

        return x, torch.flatten(emb, start_dim=1)

class Autoencoder_2(nn.Module):

    def __init__(self, input_sizes=(64, 16), channel_sequence=[1, 8, 16, 32, 64]):
        super(Autoencoder_2, self).__init__()
        self.channel_sequence = channel_sequence
        self.k = len(self.channel_sequence[:-1])
        self.H, self.W = input_sizes
        self.unprojected_dim = (self.H // 2**self.k) * (self.W // 2**self.k) * self.channel_sequence[-1]

        self._build_encoder()
        self._build_decoder()

    def _build_encoder(self):
        self.encoder = nn.Sequential()
        for i in range(len(self.channel_sequence[:-1])):
            self.encoder.append(
                nn.Conv2d(in_channels=self.channel_sequence[i],
                          out_channels=self.channel_sequence[i + 1],
                          kernel_size=3, stride=1, padding=1))
            self.encoder.append(nn.BatchNorm2d(num_features=self.channel_sequence[i + 1]))
            self.encoder.append(nn.ReLU())
            if self.channel_sequence[i] <= 16:
                self.encoder.append(nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True))

    def _build_decoder(self):
        self.decoder = nn.Sequential()
        reversed = self.channel_sequence[::-1]
        for i in range(len(reversed[:-2])):
            if reversed[i] <= 32:
                self.decoder.append(nn.MaxUnpool2d(kernel_size=2, stride=2))
            self.decoder.append(
                nn.ConvTranspose2d(in_channels=reversed[i],
                                   out_channels=reversed[i + 1],
                                   kernel_size=3, stride=1, padding=1))
            self.decoder.append(nn.BatchNorm2d(reversed[i + 1]))
            self.decoder.append(nn.ReLU())
        self.decoder.append(nn.MaxUnpool2d(kernel_size=2, stride=2))
        self.decoder.append(nn.ConvTranspose2d(in_channels=reversed[-2],
                                               out_channels=reversed[-1],
                                               kernel_size=(3, 3), stride=1, padding=1))

    def forward(self, x):
        indices_list = []
        for layer in self.encoder:
            if isinstance(layer, nn.MaxPool2d):
                x, indices = layer(x)
                indices_list.append(indices)
            else:
                x = layer(x)
        emb = x
        for layer in self.decoder:
            if isinstance(layer, nn.MaxUnpool2d):
                x = layer(x, indices_list[-1])
                indices_list = indices_list[:-1]
            else:
                x = layer(x)
        return x, torch.flatten(emb, start_dim=1)

