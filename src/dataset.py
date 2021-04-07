"""Pascal VOC Dataset Segmentation Dataloader"""

# Import modules
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image


# Define class sequences
VOC_CLASSES = ('background',  # always index 0
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')

# Define number of classes
NUM_CLASSES = len(VOC_CLASSES) + 1



# Define Dataset class
class PascalVOCDataset(Dataset):
    """Pascal VOC 2007 Dataset"""
    def __init__(self, list_file, img_dir, mask_dir, transform=None):
        # Read and parse image list
        self.images = open(list_file, "rt").read().split("\n")[:-1]
        # Initialize used transform
        self.transform = transform

        # Define input image extension
        self.img_extension = ".jpg"
        # Define output image extension
        self.mask_extension = ".png"

        # Initialize input image directory
        self.image_root_dir = img_dir
        # Initialize target directory
        self.mask_root_dir = mask_dir

        # Get class probability
        self.counts = self.__compute_class_probability()

    # Define Dataset length method
    def __len__(self):
        # Return number of input images
        return len(self.images)

    # Define getting item method from dataset
    def __getitem__(self, index):
        # Get image name
        name = self.images[index]
        # Make absolute image path
        image_path = os.path.join(self.image_root_dir, name + self.img_extension)
        # Make absolute target path
        mask_path = os.path.join(self.mask_root_dir, name + self.mask_extension)

        # Load image
        image = self.load_image(path=image_path)
        # Load target
        gt_mask = self.load_mask(path=mask_path)

        # Make data dictionary
        data = {
                    'image': torch.FloatTensor(image),
                    'mask' : torch.LongTensor(gt_mask)
                    }

        # Return data
        return data

    # Define computing method of all class counts
    def __compute_class_probability(self):
        # Initialize dictionary which has each class counts
        counts = dict((i, 0) for i in range(NUM_CLASSES))

        # Iterate images
        for name in self.images:
            # Make target path
            mask_path = os.path.join(self.mask_root_dir, name + self.mask_extension)

            # Read raw image
            raw_image = Image.open(mask_path).resize((224, 224))
            # Reshape raw image
            imx_t = np.array(raw_image).reshape(224*224)
            # Calculate each counts
            imx_t[imx_t==255] = len(VOC_CLASSES)

            # Calculate all counts
            for i in range(NUM_CLASSES):
                counts[i] += np.sum(imx_t == i)

        return counts

    # Define computing method of all class probabiliry
    def get_class_probability(self):
        # Calculate class probability from counts
        values = np.array(list(self.counts.values()))
        p_values = values/np.sum(values)

        # Return probabilities on Tensor
        return torch.Tensor(p_values)

    # Define method to load image
    def load_image(self, path=None):
        # Read image
        raw_image = Image.open(path)
        # Transpose image
        raw_image = np.transpose(raw_image.resize((224, 224)), (2,1,0))
        # Divide pixel values by max value
        imx_t = np.array(raw_image, dtype=np.float32)/255.0

        return imx_t

    # Define method to load target image
    def load_mask(self, path=None):
        # Read target image
        raw_image = Image.open(path)
        # Resize image
        raw_image = raw_image.resize((224, 224))
        # Change image to numpy array
        imx_t = np.array(raw_image)
        # Set class counts
        imx_t[imx_t==255] = len(VOC_CLASSES)

        return imx_t


if __name__ == "__main__":
    # Set data root directory path
    data_root = os.path.join("data", "VOCdevkit", "VOC2007")
    # Set train list file path
    list_file_path = os.path.join(data_root, "ImageSets", "Segmentation", "train.txt")
    # Set input image path
    img_dir = os.path.join(data_root, "JPEGImages")
    # Set target image path
    mask_dir = os.path.join(data_root, "SegmentationObject")


    # Initialize Dataset
    objects_dataset = PascalVOCDataset(list_file=list_file_path,
                                       img_dir=img_dir,
                                       mask_dir=mask_dir)

    # Get class probabilities
    print(objects_dataset.get_class_probability())

    # Get sample data
    sample = objects_dataset[0]
    image, mask = sample['image'], sample['mask']

    # Transpose image
    image.transpose_(0, 2)

    # Initilalize plt figure
    fig = plt.figure()

    # Visualize input image and target image
    a = fig.add_subplot(1,2,1)
    plt.imshow(image)

    a = fig.add_subplot(1,2,2)
    plt.imshow(mask)

    plt.show()

