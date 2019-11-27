class Config():
    def __init__(self):
        self.input_size = 224 # 224x224 pixels
        self.batch_size = 24 # Batch size of patches
        self.k = 10 # Number of classes
        self.num_epochs = 50
        self.data_dir = "./datasets/BSDS300/images/" # Directory of images
        self.showdata = False # Debug the data augmentation by showing the data we're training on.
        self.cat_dim = 0 # Feature channel dimension
