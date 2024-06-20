import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import torch.nn as nn
import torch.nn.functional as F

class ConvolutionBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(ConvolutionBlock, self).__init__()
        self.convolution = nn.Conv2d(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     bias=False)
        self.batchnorm2d = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.convolution(x)
        x = self.batchnorm2d(x)
        return self.relu(x)


class SimpleCNN(nn.Module):

    def __init__(self, channel_sequence=[1, 8, 16, 32, 64]):
        super(SimpleCNN, self).__init__()
        modules = []
        for i, c in enumerate(channel_sequence[:-1]):
            modules.append(
                ConvolutionBlock(in_channels=c,
                                 out_channels=channel_sequence[i + 1],
                                 kernel_size=3,
                                 padding=1,
                                 stride=1))
            modules.append(nn.MaxPool2d(kernel_size=2, stride=2))

        modules.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        modules.append(nn.Flatten())

        self.encoder = nn.Sequential(*modules)

    def forward(self, x):
        x = self.encoder(x)

        return F.normalize(x, p=2)
