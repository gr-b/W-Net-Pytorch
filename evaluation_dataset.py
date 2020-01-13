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

# Assumes data directory/mode has a directory called "images" and one called "segmentations"
# Loads image as input, segmentation as output
# Transforms are specified in this file
class EvaluationDataset(Dataset):
    def __init__(self, mode):
        self.mode = mode # The "test" directory name
        self.data_path  = os.path.join(config.data_dir, mode)
        self.images_dir = os.path.join(self.data_path, 'images')
        self.seg_dir    = os.path.join(self.data_path, 'segmentations')
        self.image_list = self.get_image_list()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, i):
        # Get the ith item of the dataset
        image_filepath, segmentation_filepath = self.image_list[i]
        image        = self.load_pil_image(image_filepath)
        segmentation = self.load_segmentation(segmentation_filepath)

        return toTensor(image), toTensor(segmentation)

    def get_image_list(self):
        image_list = []
        for file in os.listdir(self.images_dir):
            if file.endswith(file_ext):
                image_path = os.path.join(self.images_dir, file)
                seg_path   = os.path.join(self.seg_dir,    file.split('.')[0]+'.seg.npy')
                image_list.append((image_path, seg_path))
        return image_list

    def load_pil_image(self, path):
    # open path as file to avoid ResourceWarning
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def load_segmentation(self, path):
        return np.load(path)
