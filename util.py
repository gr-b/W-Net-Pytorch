import os, shutil
from config import Config
import torch
from torchvision import transforms

config = Config()

randomCrop = transforms.RandomCrop(config.input_size)
centerCrop = transforms.CenterCrop(config.input_size)
toTensor   = transforms.ToTensor()
toPIL      = transforms.ToPILImage()

# Clear progress images directory
def clear_progress_dir():
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


def transform_to_expected(inputs):
    inputs = [toPIL(img) for img in inputs]
    outputs_expected = None
    if config.variationalTranslation > 0:
        outputs_expected = torch.stack([toTensor(randomCrop(img)) for img in inputs]).cuda()
    inputs = torch.stack([toTensor(centerCrop(img)) for img in inputs])
    return inputs, outputs_expected
