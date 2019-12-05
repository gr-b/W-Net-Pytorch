class Config():
    def __init__(self):
        self.input_size = 224 # 224x224 pixels
        self.batch_size = 5 # Batch size of patches Note: 11 gig gpu will max batch of 5
        self.k = 10 # Number of classes
        self.num_epochs = 5#250 for real
        self.data_dir = "./datasets/BSDS300/images/" # Directory of images
        self.showdata = False # Debug the data augmentation by showing the data we're training on.
        self.cat_dim = 1 # Feature channel dimension
        self.useBatchNorm = False
