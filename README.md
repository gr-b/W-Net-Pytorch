# W-Net: A Deep Model for Fully Unsupervised Image Segmentation - Implementation in Pytorch
Unsupervised Semantic Segmentation - Bottleneck autoencoder architecture with soft n-cut loss

## Directory Structure
After cloning the repository, your dataset images should go in the following directory.
`./datasets/<dataset_name>/train/images`
For evaluation, put images and segmentations in folders:
`./datasets/<dataset_name>/test/images`
`./datasets/<dataset_name>/test/segmentations`

Each image filename and segmentation filename should correspond.

Make sure the `./datasets/<dataset_name>/` directory is a pointed to in the `config.py` file.

## Dependencies
1. Python 3
2. pytorch, torchvision, matplotlib

## Training
After setting up the dataset, you can run `python3 train.py`. You may want to change the batch size in `config.py`. A single (224x224) image patch takes a bit over 2 gigs of memory.

## Notes on this implementation
- LayerNorm used instead of BatchNorm so that a smaller batch size can be used.
- The patch size used for training and inference is set in the configuration.
  To preserve the aspect ratio seen by the model:
  The original image is first randomly cropped to 224x224 (take a random patch),
  then this is resized down to 128x128 (preserving aspect ratio),
  then finally, a random patch of size 96x96 is taken within the 128x128.
