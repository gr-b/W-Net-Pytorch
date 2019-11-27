import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

config = Config()

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

''' Each module consists of two 3 x 3 conv layers, each followed by a ReLU
non-linearity and batch normalization.

In the expansive path, modules are connected via transposed 2D convolution
layers.

The input of each module in the contracting path is also bypassed to the
output of its corresponding module in the expansive path

'''

# NOTE: introducing skip-connections increases the number of channels into each module (I think)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # TODO: Padding so that it stays as 224,112,etc
        # TODO: Add batch norm after each relu

        # Contracting path
        self.module2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3),  # TODO: Make this conv separable
            nn.ReLU(),
            nn.Conv2d(128, 128, 3), # TODO: Make this conv separable
            nn.ReLU(),
        )

        self.module3 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3), # TODO: Make this conv separable
            nn.ReLU(),
            nn.Conv2d(256, 256, 3), # TODO: Make this conv separable
            nn.ReLU(),
        )

        self.module4 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3), # TODO: Make this conv separable
            nn.ReLU(),
            nn.Conv2d(512, 512, 3), # TODO: Make this conv separable
            nn.ReLU(),
        )

        self.module5 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 1024, 3), # TODO: Make this conv separable
            nn.ReLU(),
            nn.Conv2d(1024, 1024, 3), # TODO: Make this conv separable
            nn.ReLU(),
        )

        #Expanding path
        self.module6 = nn.Sequential(
            nn.ConvTranspose2d(1024, 1024, 2), # Upconv --> Do we need to put anything after this?
            nn.Conv2d(1024, 512, 3), # TODO: Make this conv separable
            nn.ReLU(),
            nn.Conv2d(512, 512, 3), # TODO: Make this conv separable
            nn.ReLU(),
        )

        self.module7 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 2), # Upconv --> Do we need to put anything after this?
            nn.Conv2d(512, 256, 3), # TODO: Make this conv separable
            nn.ReLU(),
            nn.Conv2d(256, 256, 3), # TODO: Make this conv separable
            nn.ReLU(),
        )

        self.module8 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 2), # Upconv --> Do we need to put anything after this?
            nn.Conv2d(256, 128, 3), # TODO: Make this conv separable
            nn.ReLU(),
            nn.Conv2d(128, 128, 3), # TODO: Make this conv separable
            nn.ReLU(),
        )


    def forward(self, x):

        return self.mean_head(x), self.std_head(x)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.module1 = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
        )

        self.Uenc = UNet()

        self.module9 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 2), # Upconv --> Do we need to put anything after this?
            nn.Conv2d(128, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, config.k, 1), # TODO: Make this conv separable
            nn.ReLU(),
            nn.Softmax(dim=1), # Is this softmax dim correct?
        )

    def forward(self, x):

        return self.mean_head(x), self.std_head(x)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.module10 = nn.Sequential(
            nn.Conv2d(config.k, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
        )

        self.Udec = UNet()

        self.module9 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 2), # Upconv --> Do we need to put anything after this?
            nn.Conv2d(128, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 3, 1), # TODO: Make this conv separable
            nn.ReLU(),
            nn.Softmax(dim=1), # Is this softmax dim correct?
        )


    def forward(self, z):

        return self.sig(x)
