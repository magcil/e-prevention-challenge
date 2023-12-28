from typing import List

import torch
import torch.nn as nn
import numpy as np


class CNN_Classifier(nn.Module):

    def __init__(self, H: int, W: int, num_classes: int, linear_layers: List[int] = [256, 128]):
        super(CNN_Classifier, self).__init__()

        self.num_classes = num_classes
        self.linear_layers = linear_layers
        # Infer embedding dimension
        self.emb_dim = (H // (2**3)) * (W // (2**3)) * 32
        self.linear_layers = [self.emb_dim] + self.linear_layers + [num_classes]

        self.encoder = nn.Sequential(nn.Conv2d(1, 8, kernel_size=(3, 3), stride=1, padding=1), nn.BatchNorm2d(8),
                                     nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
                                     nn.Conv2d(8, 16, kernel_size=(3, 3), stride=1, padding=1), nn.BatchNorm2d(16),
                                     nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
                                     nn.Conv2d(16, 32, kernel_size=(3, 3), stride=1, padding=1), nn.BatchNorm2d(32),
                                     nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2), nn.Flatten())

        self.classifer = nn.Sequential()
        for i in range(len(self.linear_layers[:-1])):

            if i != len(self.linear_layers[:-1]) - 1:
                self.classifer.append(
                    nn.Linear(in_features=self.linear_layers[i], out_features=self.linear_layers[i + 1]))
                self.classifer.append(nn.BatchNorm1d(num_features=self.linear_layers[i + 1]))
                self.classifer.append(nn.ReLU())
            else:
                self.classifer.append(
                    nn.Linear(in_features=self.linear_layers[i], out_features=self.linear_layers[i + 1]))
    
    def forward(self, x):
        emb = self.encoder(x)

        return self.classifer(emb), emb
