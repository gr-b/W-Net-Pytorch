# Converts Berkeley segmentation dataset segmentation format files to .npy arrays
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

destination = "./datasets/segmentations"

def convertAndSave(filepath, filename):
    f = open(filepath, 'r')
    w, h = (0, 0)
    for line in f:
        if 'width' in line:
            w = int(line.split(' ')[1])
        if 'height' in line:
            h = int(line.split(' ')[1])
        if 'data' in line:
            break

    seg = np.zeros((h, w))
    for line in f:
        s, r, c1, c2 = map(lambda x: int(x), line.split(' '))
        seg[r, c1:c2] = s

    #filename = filename + ".png"
    path = os.path.join(destination, filename)
    np.save(path, seg)
    #matplotlib.image.imsave(path, seg) # Saves but pixels values in [0.1]


path = "./datasets/human/color/"
dirs = list()
for dir, _, files in os.walk(path):
    for filename in files:
        filepath = os.path.join(dir, filename)
        print(filepath)
        convertAndSave(filepath, filename)
