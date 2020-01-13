# Model evaluation script
# As the model is unsupervised, this script tries all possible mappings
# Of segmentation to label and takes the best one.

# Since the model works on patches, we don't need to do any transforms, instead, we just
# need to cut the image into patches and feed all of them through
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

from config import Config
import util
from model import WNet
from evaluation_dataset import EvaluationDataset

def main():
    print("PyTorch Version: ",torch.__version__)
    if torch.cuda.is_available():
        print("Cuda is available. Using GPU")

    config = Config()

    ###################################
    # Image loading and preprocessing #
    ###################################

    evaluation_dataset = EvaluationDataset("test")

    evaluation_dataloader = torch.utils.data.DataLoader(evaluation_dataset,
        batch_size=config.test_batch_size, num_workers=4, shuffle=False)

    ###################################
    #          Model Setup            #
    ###################################

    # We will only use .forward_encoder()
    autoencoder = torch.load("./models/model", map_location=torch.device('cpu'))
    util.enumerate_params([autoencoder])

    ###################################
    #          Training Loop          #
    ###################################

    autoencoder.eval()

    def combine_patches(image, patches):
        w, h = image[0].shape
        segmentation = torch.zeros(w, h)
        x, y = (0,0) # Start of next patch
        for patch in patches:
            if y+size > h:
                y = 0
                x += size
            segmentation[x:x+size,y:y+size] = patch
            y += size
        return segmentation


    for i, [images, segmentations] in enumerate(evaluation_dataloader, 0):
        size = config.input_size
        for image, seg in zip(images, segmentations):
            # NOTE: problem - the above won't get all patches, only ones that fit.
            patches = image.unfold(0, 3, 3).unfold(1, size, size).unfold(2, size, size)
        patch_batch = patches.reshape(-1, 3, size, size)

        seg_batch = autoencoder.forward_encoder(patch_batch)
        seg_batch = torch.argmax(seg_batch, axis=1).float()

        segmentation = combine_patches(image, seg_batch)

        f, axes = plt.subplots(1, 2, figsize=(8,8))
        axes[0].imshow(segmentation)
        axes[1].imshow(image.permute(1, 2, 0))
        plt.show()



if __name__ == "__main__":
    main()
