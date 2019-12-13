from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import numpy as np
import glob

from config import Config

config = Config()

file_ext = ".jpg"

# NOTE: We can't precompute the image weights because of the data augmentation (the crop)

# sample of our data is a dict: {'image': image, 'weight': weights}

# Load the image and the weight matrix for that image.
# The weight matrix w is a measure of the weight between each pixel and
# every other pixel. so w[u][v] is a measure of
#   (a) Distance between the brightness of the two pixels.
#   (b) Distance in positon between the two pixels

# Assumes images_dir (train directory) has a directory called "images"
class NCutWeightDataset(Dataset):
    def __init__(self, images_dir, composed_transforms):
        self.raw_data = []
        self.images_dir = os.path.join(images_dir, 'images')
        self.transforms = composed_transforms

        self.toGrayscale = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()])

        self.image_list = self.get_image_list()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, i):
        # Get the ith item of the dataset
        filepath = self.image_list[i]
        image = self.load_pil_image(filepath)
        image = self.transforms(image)

        return (image, self.calculate_connection_weights(image))

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

    def calculate_connection_weights(self, image):
        # Given 3x224x224 tensor image,
        image = self.toGrayscale(image)
        return image
