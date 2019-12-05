import torch
import torch.nn as nn
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

# Padding=1 because (3x3) conv leaves of 2pixels in each dimension, 1 on each side
# Do we want non-linearity between pointwise and depthwise (separable) conv?
# Do we want non-linearity after upconv?

# Note: I manually make each conv layer take half as many channels to accommodate the skip-connections
# As opposed to convolving down the number of channels in the forward pass first. I think this is how it is intended

class ConvModule(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConvModule, self).__init__()

        layers = [
            nn.Conv2d(input_dim, output_dim, 1), # Pointwise (1x1) through all channels
            nn.Conv2d(output_dim, output_dim, 3, padding=1, groups=output_dim), # Depthwise (3x3) through each channel
            nn.ReLU(),
            nn.BatchNorm2d(output_dim),
            nn.Conv2d(output_dim, output_dim, 1),
            nn.Conv2d(output_dim, output_dim, 3, padding=1, groups=output_dim),
            nn.ReLU(),
            nn.BatchNorm2d(output_dim),
        ]

        if not config.useBatchNorm:
            layers = [layer for layer in layers if not isinstance(layer, nn.BatchNorm2d)]

        self.module = nn.Sequential(*layers)

    def forward(self, x):
        return self.module(x)

class BaseNet(nn.Module): # 1 U-net
    def __init__(self, input_channels, output_channels):
        super(BaseNet, self).__init__()

        layers = [
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        ]

        if not config.useBatchNorm:
            layers = [layer for layer in layers if not isinstance(layer, nn.BatchNorm2d)]

        self.module1 = nn.Sequential(*layers)


        self.module12pool = nn.MaxPool2d(2, 2)
        self.module2 = ConvModule(64, 128)
        self.module23pool = nn.MaxPool2d(2, 2)
        self.module3 = ConvModule(128, 256)
        self.module34pool = nn.MaxPool2d(2, 2)
        self.module4 = ConvModule(256, 512)
        self.module45pool = nn.MaxPool2d(2, 2)
        self.module5 = ConvModule(512, 1024)

        self.module56upconv = nn.ConvTranspose2d(1024, 1024, 2, stride=2) # Stride of 2 makes it the right size
        self.module6 = ConvModule(1024+512, 512)
        self.module67upconv = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.module7 = ConvModule(512+256, 256)
        self.module78upconv = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.module8 = ConvModule(256+128, 128)

        self.module89upconv = nn.ConvTranspose2d(128, 128, 2, stride=2)


        layers = [
            nn.Conv2d(128+64, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, output_channels, 1), # No padding on pointwise
            nn.ReLU(),
        ]

        if not config.useBatchNorm:
            layers = [layer for layer in layers if not isinstance(layer, nn.BatchNorm2d)]

        self.module9 = nn.Sequential(*layers)


        self.softmax = nn.Softmax2d()


    def forward(self, x):
        x1 = self.module1(x)
        x2 = self.module2(self.module12pool(x1))
        x3 = self.module3(self.module23pool(x2))
        x4 = self.module4(self.module34pool(x3))
        x5 = self.module5(self.module45pool(x4))

        x6 = self.module6(
            torch.cat((x4, self.module56upconv(x5)), config.cat_dim)
        )
        x7 = self.module7(
            torch.cat((x3, self.module67upconv(x6)), config.cat_dim)
        )
        x8 = self.module8(
            torch.cat((x2, self.module78upconv(x7)), config.cat_dim)
        )
        x9 = self.module9(
            torch.cat((x1, self.module89upconv(x8)), config.cat_dim)
        )

        segmentations = self.softmax(x9)
        return segmentations


class WNet(nn.Module):
    def __init__(self):
        super(WNet, self).__init__()

        self.U_encoder = BaseNet(input_channels=3, output_channels=config.k)
        self.softmax = nn.Softmax2d()
        self.U_decoder = BaseNet(input_channels=config.k, output_channels=3)

    def forward_encoder(self, x):
        x9 = self.U_encoder(x)
        segmentations = self.softmax(x9)
        return segmentations

    def forward_decoder(self, segmentations):
        x18 = self.U_decoder(segmentations)
        return x18

    def forward(self, x): # x is (3 channels 224x224)
        segmentations = self.forward_encoder(x)
        x_prime       = self.forward_decoder(segmentations)
        return segmentations, x_prime
