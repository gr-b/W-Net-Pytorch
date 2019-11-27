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

we double the number of feature channels at each downsampling step
We halve the number of feature channels at each upsampling step

'''

# NOTE: batch norm is up for debate

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
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3), # TODO: Make this conv separable
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )

        self.module3 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3), # TODO: Make this conv separable
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3), # TODO: Make this conv separable
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )

        self.module4 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3), # TODO: Make this conv separable
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3), # TODO: Make this conv separable
            nn.ReLU(),
            nn.BatchNorm2d(512),
        )

        self.module5 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 1024, 3), # TODO: Make this conv separable
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 1024, 3), # TODO: Make this conv separable
            nn.ReLU(),
            nn.BatchNorm2d(1024),
        )

        #Expanding path
        self.module6 = nn.Sequential(
            nn.ConvTranspose2d(1024, 1024, 2), # Upconv --> Do we need to put anything after this?
            nn.Conv2d(1024, 512, 3), # TODO: Make this conv separable
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3), # TODO: Make this conv separable
            nn.ReLU(),
            nn.BatchNorm2d(512),
        )

        self.module7 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 2), # Upconv --> Do we need to put anything after this?
            nn.Conv2d(512, 256, 3), # TODO: Make this conv separable
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3), # TODO: Make this conv separable
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )

        self.module8 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 2), # Upconv --> Do we need to put anything after this?
            nn.Conv2d(256, 128, 3), # TODO: Make this conv separable
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3), # TODO: Make this conv separable
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 2),
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
            nn.Conv2d(128, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, config.k, 1), # TODO: Make this conv separable
            nn.ReLU(),
            nn.Softmax2d(dim=0), # Is this softmax dim correct? (want to softmax across K dimension)
        )

    def forward(self, x): # x is (3 channels 224x224)
        module1_out = self.module1(x) # (64x224x224)
        u_enc_out = self.Uenc(module1out) # (64x112x112)

        # Add in Skip-connection
        module9in = torch.cat((module1_out, u_enc_out), 0) # concat along channel dimension (0)

        segmentations = self.module9(module9in) # (K x 224 x 224)
        return segmentations

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
