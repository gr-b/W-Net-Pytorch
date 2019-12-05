# Implementation of W-Net: A Deep Model for Fully Unsupervised Image Segmentation
# in Pytorch.
# Author: Griffin Bishop

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os, shutil
import copy
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

from config import Config
import util
from model import WNet

config = Config()

# NOTE: We're not currently using this variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

###################################
# Image loading and preprocessing #
###################################

# For now, data augmentation must not introduce any missing pixels
train_xform = transforms.Compose([
    transforms.RandomCrop(config.input_size), # For now, cropping down to 224
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
val_xform = transforms.Compose([
    transforms.RandomCrop(config.input_size), # For now, cropping down to 224
    transforms.ToTensor()
])

#TODO: Load segmentation maps too
train_dataset = datasets.ImageFolder(os.path.join(config.data_dir, "train"), train_xform)
val_dataset   = datasets.ImageFolder(os.path.join(config.data_dir, "test"),  val_xform)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, num_workers=4, shuffle=True)
val_dataloader   = torch.utils.data.DataLoader(val_dataset,   batch_size=config.batch_size, num_workers=4, shuffle=True)

util.clear_progress_dir()

###################################
#          Model Setup            #
###################################

autoencoder = WNet().cuda()
optimizer = torch.optim.Adam(autoencoder.parameters())
print(autoencoder)
util.enumerate_params([autoencoder])


###################################
#          Loss Criterion         #
###################################

def reconstruction_loss(x, x_prime):
	binary_cross_entropy = F.binary_cross_entropy(x_prime, x, reduction='sum')
	return binary_cross_entropy

#TODO: Implement soft n-cut loss
def soft_n_cut_loss(segmentations):
    return 0


###################################
#          Training Loop          #
###################################

autoencoder.train()

for epoch in range(config.num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data
        if config.showdata:
            plt.imshow(inputs[0].permute(1, 2, 0))
            plt.show()

        inputs = inputs.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()

        segmentations, reconstructions = autoencoder(inputs)

        l_soft_n_cut     = soft_n_cut_loss(segmentations)
        l_reconstruction = reconstruction_loss(inputs, reconstructions)

        loss = (l_reconstruction + l_soft_n_cut)
        loss.backward(retain_graph=True)
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        if config.showSegmentationProgress and i == 0: # If it's the first batch in the epoch
            #import ipdb; ipdb.set_trace()
            # Get the first example from the batch.

            segmentation = segmentations[0]
            pixels = torch.argmax(segmentation, axis=0) / config.k # to [0,1]
            plt.imshow(pixels.detach().cpu()) # TODO: show image on a separate thread
            plt.show()

    epoch_loss = running_loss / len(train_dataloader.dataset)
    print(f"Epoch {epoch} loss: {epoch_loss:.6f}")
