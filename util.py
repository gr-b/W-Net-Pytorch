import os, shutil
from config import Config
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

config = Config()

# Clear progress images directory
def clear_progress_dir(): # Or make the dir if it does not exist
    if not os.path.isdir(config.segmentationProgressDir):
        os.mkdir(config.segmentationProgressDir)
    else: # Clear the directory
        for filename in os.listdir(config.segmentationProgressDir):
            filepath = os.path.join(config.segmentationProgressDir, filename)
            os.remove(filepath)

def enumerate_params(models):
	num_params = 0
	for model in models:
		for param in model.parameters():
			if param.requires_grad:
				num_params += param.numel()
	print(f"Total trainable model parameters: {num_params}")

def save_progress_image(autoencoder, progress_images, epoch):
    if not torch.cuda.is_available():
        segmentations, reconstructions = autoencoder(progress_images)
    else:
        segmentations, reconstructions = autoencoder(progress_images.cuda())

    # Get the first example from the batch.
    segmentation = segmentations[0]
    pixels = torch.argmax(segmentation, axis=0).float() / config.k # to [0,1]

    f, axes = plt.subplots(4, 1, figsize=(8,8))
    axes[0].imshow(progress_images[0].permute(1, 2, 0))
    axes[1].imshow(pixels.detach().cpu())
    axes[2].imshow(reconstructions[0].detach().cpu().permute(1, 2, 0))
    if config.variationalTranslation:
        axes[3].imshow(progress_expected[0].detach().cpu().permute(1, 2, 0))
    plt.savefig(os.path.join(config.segmentationProgressDir, str(epoch)+".png"))
    plt.close(f)
