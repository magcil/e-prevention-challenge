# Convolutional Autoencoder

"""import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Define the autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 
                               kernel_size=3, 
                               stride=3, 
                               padding=1, 
                               output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 
                               kernel_size=3, 
                               stride=2, 
                               padding=1, 
                               output_padding=1),
            nn.ReLU(),
            nn.Linear(16, 15),
            nn.Sigmoid()
        )
         
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, window_size):
        super(Autoencoder, self).__init__()
        self.ws = window_size
        self.input_size = int(self.ws) * 13

        print('self input size:', self.input_size)
        
        # Encoder
        self.conv1 = nn.Conv1d(self.input_size, self.input_size//2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(self.input_size//2, self.input_size//4, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(self.input_size//4, self.input_size//10, kernel_size=3, stride=1, padding=1)
        
        # Decoder
        self.deconv1 = nn.ConvTranspose1d(self.input_size//10, self.input_size//5, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose1d(self.input_size//5, self.input_size//2, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose1d(self.input_size//2, self.input_size, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        # Encoder
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Decoder
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        
        return x
