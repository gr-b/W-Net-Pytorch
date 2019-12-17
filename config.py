class Config():
    def __init__(self):
        self.debug = True
        self.input_size = 96 # Side length of square image patch
        self.batch_size = 1 # Batch size of patches Note: 11 gig gpu will max batch of 5
        self.k = 4 # Number of classes
        self.num_epochs = 250#250 for real
        self.data_dir = "./datasets/BSDS300/images/" # Directory of images
        self.showdata = False # Debug the data augmentation by showing the data we're training on.

        self.useBatchNorm = True
        self.useDropout = True

        self.showSegmentationProgress = True
        self.segmentationProgressDir = './latent_images/'
        self.epochShuffle = True
        # False if we want to see image progress (but then SGD doesn't work right;
        # same batches every time)
        self.variationalTranslation = 0 # Pixels, 0 for off. 1 works fine
