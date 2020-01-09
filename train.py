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
from datetime import datetime
import os, shutil
import copy

from config import Config
import util
from model import WNet
from autoencoder_dataset import AutoencoderDataset
from soft_n_cut_loss import soft_n_cut_loss

def main():
    print("PyTorch Version: ",torch.__version__)
    if torch.cuda.is_available():
        print("Cuda is available. Using GPU")

    config = Config()

    ###################################
    # Image loading and preprocessing #
    ###################################

    #TODO: Maybe we should crop a large square, then resize that down to our patch size?
    # For now, data augmentation must not introduce any missing pixels TODO: Add data augmentation noise
    train_xform = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.Resize(128),
        transforms.RandomCrop(config.input_size+config.variationalTranslation), # For now, cropping down to 224
        transforms.RandomHorizontalFlip(), # TODO: Add colorjitter, random erasing
        transforms.ToTensor()
    ])
    val_xform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.Resize(128),
        transforms.CenterCrop(config.input_size),
        transforms.ToTensor()
    ])

    #TODO: Load validation segmentation maps too  (for evaluation purposes)
    train_dataset = AutoencoderDataset("train", train_xform)
    val_dataset   = AutoencoderDataset("val", val_xform)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, num_workers=4, shuffle=True)
    val_dataloader   = torch.utils.data.DataLoader(val_dataset,   batch_size=4, num_workers=4, shuffle=False)

    util.clear_progress_dir()

    ###################################
    #          Model Setup            #
    ###################################

    autoencoder = WNet()
    if torch.cuda.is_available():
        autoencoder = autoencoder.cuda()
    optimizer = torch.optim.Adam(autoencoder.parameters())
    if config.debug:
        print(autoencoder)
    util.enumerate_params([autoencoder])

    # Use the current time to save the model at end of each epoch
    modelName = str(datetime.now())



    ###################################
    #          Loss Criterion         #
    ###################################

    def reconstruction_loss(x, x_prime):
    	binary_cross_entropy = F.binary_cross_entropy(x_prime, x, reduction='sum')
    	return binary_cross_entropy


    ###################################
    #          Training Loop          #
    ###################################

    autoencoder.train()

    progress_images, progress_expected = next(iter(val_dataloader))

    for epoch in range(config.num_epochs):
        running_loss = 0.0
        for i, [inputs, outputs] in enumerate(train_dataloader, 0):

            if config.showdata:
                print(inputs.shape)
                print(outputs.shape)
                print(inputs[0])
                plt.imshow(inputs[0].permute(1, 2, 0))
                plt.show()

            if torch.cuda.is_available():
                inputs  = inputs.cuda()
                outputs = outputs.cuda()

            optimizer.zero_grad()

            segmentations, reconstructions = autoencoder(inputs)

            l_soft_n_cut     = soft_n_cut_loss(inputs, segmentations)
            l_reconstruction = reconstruction_loss(
                inputs if config.variationalTranslation == 0 else outputs,
                reconstructions
            )

            loss = (l_reconstruction + l_soft_n_cut)
            loss.backward(retain_graph=False) # We only need to do retain graph =true if we're backpropping from multiple heads
            optimizer.step()

            if config.debug and (i%50) == 0:
                print(i)

            # print statistics
            running_loss += loss.item()


            if config.showSegmentationProgress and i == 0: # If first batch in epoch
                util.save_progress_image(autoencoder, progress_images, epoch)
                optimizer.zero_grad() # Don't change gradient on validation

        epoch_loss = running_loss / len(train_dataloader.dataset)
        print(f"Epoch {epoch} loss: {epoch_loss:.6f}")

        if config.saveModel:
            util.save_model(autoencoder, modelName)



if __name__ == "__main__":
    main()
