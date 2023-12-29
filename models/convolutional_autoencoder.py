# Convolutional Autoencoder
import torch
import torch.nn as nn
import numpy as np


# Define the autoencoder architecture
class Autoencoder(nn.Module):

    def __init__(self):
        super(Autoencoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 8, kernel_size=(3, 3), stride=1, padding=1), nn.BatchNorm2d(4),
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

        return x, torch.flatten(x, start_dim=1)


class Autoencoder_2(nn.Module):

    def __init__(self, input_sizes=(64, 16), channel_sequence=[1, 8, 16, 32, 64], latent_dim=128):
        super(Autoencoder_2, self).__init__()
        self.channel_sequence = channel_sequence
        self.k = len(self.channel_sequence[:-1])
        self.latent_dim = latent_dim
        self.H, self.W = input_sizes
        self.unprojected_dim = (self.H // 2**self.k) * (self.W // 2**self.k) * self.channel_sequence[-1]

        self._build_encoder()
        self._build_bottleneck()
        self._build_decoder()

    def _build_encoder(self):
        self.encoder = nn.Sequential()
        for i in range(len(self.channel_sequence[:-1])):
            self.encoder.append(
                nn.Conv2d(in_channels=self.channel_sequence[i],
                          out_channels=self.channel_sequence[i + 1],
                          kernel_size=3,
                          padding="same",
                          bias=False))
            self.encoder.append(nn.BatchNorm2d(num_features=self.channel_sequence[i + 1]))
            self.encoder.append(nn.ReLU())
            self.encoder.append(nn.MaxPool2d(kernel_size=2, stride=2))

    def _build_decoder(self):
        self.decoder = nn.Sequential()
        reversed = self.channel_sequence[::-1]
        for i in range(len(reversed[:-1])):
            self.decoder.append(
                nn.ConvTranspose2d(in_channels=reversed[i], out_channels=reversed[i], kernel_size=2, stride=2))
            self.decoder.append(nn.BatchNorm2d(reversed[i]))
            self.decoder.append(nn.ReLU())
            self.decoder.append(
                nn.ConvTranspose2d(in_channels=reversed[i],
                                   out_channels=reversed[i + 1],
                                   kernel_size=3,
                                   padding=1,
                                   bias=False))

    def _build_bottleneck(self):
        self.projection = nn.Sequential(nn.Flatten(),
                                        nn.Linear(in_features=self.unprojected_dim, out_features=self.latent_dim))
        self.inv_projection = nn.Sequential(
            nn.Linear(in_features=self.latent_dim, out_features=self.unprojected_dim),
            nn.Unflatten(dim=1, unflattened_size=(self.channel_sequence[-1], self.H // 2**self.k, self.W // 2**self.k)))

    def forward(self, x):
        x = self.encoder(x)

        emb = self.projection(x)

        return self.decoder(self.inv_projection(emb))


# Unet Architecture
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
                                  nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):

    def __init__(self, in_channels, out_channels, features=[16, 32, 64]):
        super(UNet, self).__init__()
        self.down_part = nn.ModuleList()
        self.up_part = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder Part
        for feature in features:
            self.down_part.append(DoubleConv(in_channels, feature))
            in_channels = feature
        # Decoder Part
        for feature in reversed(features):
            self.up_part.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.up_part.append(DoubleConv(2 * feature, feature))
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.output = nn.Conv2d(features[0], out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        skip_connections = []
        for down in self.down_part:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.up_part), 2):
            x = self.up_part[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.up_part[idx + 1](concat_skip)

        return self.output(x)
