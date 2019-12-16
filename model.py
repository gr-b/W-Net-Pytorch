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

        self.first_module = nn.Sequential(*layers)

        self.pool = nn.MaxPool2d(2, 2)
        encoder_in_sizes = [64, 128, 256, 512]
        self.enc_modules = nn.ModuleList(
            [ConvModule(channels, 2*channels) for channels in encoder_in_sizes])

        decoder_in_sizes = [1024, 512, 256]
        decoder_out_sizes = [512, 256, 128]
        self.dec_transpose_layers = nn.ModuleList(
            [nn.ConvTranspose2d(channels, channels, 2, stride=2) for channels in decoder_in_sizes]) # Stride of 2 makes it right size
        self.dec_modules = nn.ModuleList(
            [ConvModule(3*channels_out, channels_out) for channels_out in decoder_out_sizes])
        self.last_dec_transpose_layer = nn.ConvTranspose2d(128, 128, 2, stride=2)

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

        self.last_module = nn.Sequential(*layers)


    def forward(self, x):
        x1 = self.first_module(x)
        activations = [x1]
        for module in self.enc_modules:
            activations.append(module(self.pool(activations[-1])))

        x_ = activations.pop(-1)

        for conv, upconv in zip(self.dec_modules, self.dec_transpose_layers):
            skip_connection = activations.pop(-1)
            x_ = conv(
                torch.cat((skip_connection, upconv(x_)), 1)
            )

        segmentations = self.last_module(
            torch.cat((activations[-1], self.last_dec_transpose_layer(x_)), 1)
        )
        return segmentations


class WNet(nn.Module):
    def __init__(self):
        super(WNet, self).__init__()

        self.U_encoder = BaseNet(input_channels=3, output_channels=config.k)
        self.softmax = nn.Softmax2d()
        self.U_decoder = BaseNet(input_channels=config.k, output_channels=3)
        self.sigmoid = nn.Sigmoid()

    def forward_encoder(self, x):
        x9 = self.U_encoder(x)
        segmentations = self.softmax(x9)
        return segmentations

    def forward_decoder(self, segmentations):
        x18 = self.U_decoder(segmentations)
        reconstructions = self.sigmoid(x18)
        return reconstructions

    def forward(self, x): # x is (3 channels 224x224)
        segmentations = self.forward_encoder(x)
        x_prime       = self.forward_decoder(segmentations)
        return segmentations, x_prime
