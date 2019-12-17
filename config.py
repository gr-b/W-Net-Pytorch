class Config():
    def __init__(self):
        self.debug = True
        self.input_size = 96 # Side length of square image patch
        self.batch_size = 1 # Batch size of patches Note: 11 gig gpu will max batch of 5
        self.k = 4 # Number of classes
        self.num_epochs = 250#250 for real
        self.data_dir = "./datasets/BSDS300/images/" # Directory of images
        self.showdata = True # Debug the data augmentation by showing the data we're training on.

        self.useNorm = True # Instance Normalization
        self.useDropout = True
        self.drop = 0.2

        self.showSegmentationProgress = True
        self.segmentationProgressDir = './latent_images/'

        self.variationalTranslation = 2 # Pixels, 0 for off. 1 works fine
