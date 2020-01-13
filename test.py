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

    autoencoder = WNet() # We will only use .forward_encoder
    if torch.cuda.is_available():
        autoencoder = autoencoder.cuda()
    util.enumerate_params([autoencoder])

    ###################################
    #          Training Loop          #
    ###################################

    autoencoder.eval()

    for i, [images, segmentations] in enumerate(evaluation_dataloader, 0):
        import ipdb; ipdb.set_trace()



if __name__ == "__main__":
    main()
