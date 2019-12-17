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
2. Pytorch

## Training
After setting up the dataset, you can run `python3 train.py`. You may want to change the batch size in `config.py`. A single (224x224) image patch takes a bit over 2 gigs of memory. If you don't have a GPU, then you can change a few lines in `train.py` to not say `.cuda()` at the end. I will make this programmatic in the future.

## Notes
- LayerNorm used instead of BatchNorm so that a smaller batch size can be used.


Note: Model suffers heavily from overfitting (checkerboarding), despite dropout
# TODO:
1. Add evaluation metrics for evaluation dataset
2. Show batch size number of images each epoch
3. Show the same image every time so that comparison is useful
