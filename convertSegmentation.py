# Converts Berkeley segmentation dataset segmentation format files to .npy arrays
import numpy as np
import matplotlib.pyplot as plt
import os

destination = "./segmentations"

def convertAndSave(filepath, filename):
    f = open(filepath, 'r')
    w, h = (0, 0)
    for line in f:
        if 'width' in line:
            w = int(line.split(' ')[1])
            #print(f"width: {w}")
        if 'height' in line:
            h = int(line.split(' ')[1])
            #print(f"height: {h}")
        if 'data' in line:
            break

    seg = np.zeros((w, h))
    for line in f:
        s, r, c1, c2 = map(lambda x: int(x), line.split(' '))
        seg[c1:c2, r] = s

    path = os.path.join(destination, filename)
    np.save(path, seg)


path = "./human/color"
dirs = list()
for dir, _, files in os.walk(path):
    for filename in files:
        filepath = os.path.join(dir, filename)
        print(filepath)
        convertAndSave(filepath, filename)
