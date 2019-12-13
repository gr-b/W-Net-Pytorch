from PIL import Image
import torch
import torch.utils.data as Data
import os
import numpy as np

from config import Config

config = Config()

file_ext = "*.jpg"

class WeightDataLoader():
    def __init__(self, images_dir, composed_transforms):
        self.raw_data = []
        self.images_dir = images_dir
        self.transforms = composed_transforms
