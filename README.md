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

## Notes
- LayerNorm used instead of BatchNorm so that a smaller batch size can be used.


NOTE: InstanceNorm might be causing problems --- 21 epochs in 30 minutes
NOTE: Model suffers heavily from overfitting (checkerboarding), despite dropout (Variational Translation fixes this)
# TODO:
1. Add evaluation metrics for evaluation dataset
2. Show batch size number of images each epoch
3. Show the same image every time so that comparison is useful
