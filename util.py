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

def save_model(autoencoder, modelName):
    path = os.path.join("./models/", modelName.replace(":", " ").replace(".", " ").replace(" ", "_"))
    torch.save(autoencoder, path)
    with open(path+".config", "a+") as f:
        f.write(str(config))
        f.close()

def save_progress_image(autoencoder, progress_images, epoch):
    if not torch.cuda.is_available():
        segmentations, reconstructions = autoencoder(progress_images)
    else:
        segmentations, reconstructions = autoencoder(progress_images.cuda())

    f, axes = plt.subplots(4, config.val_batch_size, figsize=(8,8))
    for i in range(config.val_batch_size):
        segmentation = segmentations[i]
        pixels = torch.argmax(segmentation, axis=0).float() / config.k # to [0,1]

        axes[0, i].imshow(progress_images[i].permute(1, 2, 0))
        axes[1, i].imshow(pixels.detach().cpu())
        axes[2, i].imshow(reconstructions[i].detach().cpu().permute(1, 2, 0))
        if config.variationalTranslation:
            axes[3, i].imshow(progress_expected[i].detach().cpu().permute(1, 2, 0))
    plt.savefig(os.path.join(config.segmentationProgressDir, str(epoch)+".png"))
    plt.close(f)
