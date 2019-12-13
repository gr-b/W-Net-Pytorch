from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import numpy as np
import glob
import time

from config import Config

config = Config()

file_ext = ".jpg"

randomCrop = transforms.RandomCrop(config.input_size)
centerCrop = transforms.CenterCrop(config.input_size)
toTensor   = transforms.ToTensor()
toPIL      = transforms.ToPILImage()

# Assumes images_dir (train directory) has a directory called "images"
# Loads image as both inputs and outputs
# Applies different transforms to both input and output
class AutoencoderDataset(Dataset):
    def __init__(self, mode, input_transforms):
        self.mode = mode
        self.data_path  = os.path.join(config.data_dir, mode)
        self.images_dir = os.path.join(self.data_path, 'images')
        self.image_list = self.get_image_list()
        self.transforms = input_transforms

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, i):
        # Get the ith item of the dataset
        filepath = self.image_list[i]
        input = self.load_pil_image(filepath)
        input = self.transforms(input)

        input = toPIL(input)
        output = input.copy()
        if self.mode is "train" and config.variationalTranslation > 0:
            output = randomCrop(input)
        input = toTensor(centerCrop(input))
        output = toTensor(output)

        return input, output

    def get_image_list(self):
        image_list = []
        for file in os.listdir(self.images_dir):
            if file.endswith(file_ext):
                path = os.path.join(self.images_dir, file)
                image_list.append(path)
        return image_list

    def load_pil_image(self, path):
    # open path as file to avoid ResourceWarning
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
