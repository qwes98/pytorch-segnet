"""
Infer segmentation results from a trained SegNet model


Usage:
python inference.py --data_root /home/SharedData/intern_sayan/PascalVOC2012/data/VOCdevkit/VOC2012/ \
                    --val_path ImageSets/Segmentation/val.txt \
                    --img_dir JPEGImages \
                    --mask_dir SegmentationClass \
                    --model_path /home/SharedData/intern_sayan/PascalVOC2012/model_best.pth \
                    --output_dir /home/SharedData/intern_sayan/PascalVOC2012/predictions \
                    --gpu 1
"""

# Import modules
from __future__ import print_function
import argparse
from dataset import PascalVOCDataset, NUM_CLASSES
import matplotlib.pyplot as plt
from model import SegNet
import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Set plt settings
plt.switch_backend('agg')
plt.axis('off')


# Constants
# Set input image channels
NUM_INPUT_CHANNELS = 3
# Set output channels
NUM_OUTPUT_CHANNELS = NUM_CLASSES

# Set mini-batch size
BATCH_SIZE = 8


# Arguments
parser = argparse.ArgumentParser(description='Validate a SegNet model')

# Get data root path
parser.add_argument('--data_root', required=True)
# Get validation data path
parser.add_argument('--val_path', required=True)
# Get input image path
parser.add_argument('--img_dir', required=True)
# Get target image path
parser.add_argument('--mask_dir', required=True)
# Get model path
parser.add_argument('--model_path', required=True)
# Get output path
parser.add_argument('--output_dir', required=True)
# Get whether to use gpu
parser.add_argument('--gpu', type=int)

# Parse arguments
args = parser.parse_args()


# Define validation function
def validate():
    # Change model to evalidation mode
    model.eval()

    # Iterate mini-batch from validation dataset
    for batch_idx, batch in enumerate(val_dataloader):
        # Read input image
        input_tensor = torch.autograd.Variable(batch['image'])
        # Read target image
        target_tensor = torch.autograd.Variable(batch['mask'])

        # If using cuda
        if CUDA:
            # Load input tensor on cuda
            input_tensor = input_tensor.cuda(GPU_ID)
            # Load target tensor on cuda
            target_tensor = target_tensor.cuda(GPU_ID)

        # Inference(pass input through model)
        predicted_tensor, softmaxed_tensor = model(input_tensor)
        # Calculate loss value from target and predicted values
        loss = criterion(predicted_tensor, target_tensor)

        # Iterate this mini batch result
        for idx, predicted_mask in enumerate(softmaxed_tensor):
            # Get target image
            target_mask = target_tensor[idx]
            # Get input image
            input_image = input_tensor[idx]

            # Initialize plt figure
            fig = plt.figure()

            # Set plt subplot to plot images
            a = fig.add_subplot(1,3,1)
            # Show input image
            plt.imshow(input_image.transpose(0, 2))
            a.set_title('Input Image')

            # Set plt subplot to plot images
            a = fig.add_subplot(1,3,2)
            # Load predicted mask on cpu
            predicted_mx = predicted_mask.detach().cpu().numpy()
            # Calculate predicted class index for visualization
            predicted_mx = predicted_mx.argmax(axis=0)
            # Show predicted result
            plt.imshow(predicted_mx)
            a.set_title('Predicted Mask')

            # Set plt subplot to plot images
            a = fig.add_subplot(1,3,3)
            # Load target image on cpu
            target_mx = target_mask.detach().cpu().numpy()
            # show target image 
            plt.imshow(target_mx)
            a.set_title('Ground Truth')

            # Save this figure
            fig.savefig(os.path.join(OUTPUT_DIR, "prediction_{}_{}.png".format(batch_idx, idx)))

            plt.close(fig)


if __name__ == "__main__":
    # Set data root path
    data_root = args.data_root
    # Set absolute validation data path 
    val_path = os.path.join(data_root, args.val_path)
    # Set absolute input image data path 
    img_dir = os.path.join(data_root, args.img_dir)
    # Set absolute target data path 
    mask_dir = os.path.join(data_root, args.mask_dir)

    # Set model path
    SAVED_MODEL_PATH = args.model_path
    # Set output directory path
    OUTPUT_DIR = args.output_dir

    # Set whether to use cuda 
    CUDA = args.gpu is not None
    # Set GPU ID
    GPU_ID = args.gpu


    # Initialize Pascal VOC Dataset for validation
    val_dataset = PascalVOCDataset(list_file=val_path,
                                   img_dir=img_dir,
                                   mask_dir=mask_dir)

    # Initialize validation dataloader
    val_dataloader = DataLoader(val_dataset,            
                                batch_size=BATCH_SIZE,  # mini-batch size
                                shuffle=True,           # Use shuffle
                                num_workers=4)          # number of cpu


    # If using cuda
    if CUDA:
        # Initialize SegNet model on gpu memory
        model = SegNet(input_channels=NUM_INPUT_CHANNELS,
                       output_channels=NUM_OUTPUT_CHANNELS).cuda(GPU_ID)

        # Set target class values on gpu memory
        class_weights = 1.0/val_dataset.get_class_probability().cuda(GPU_ID)
        # Set loss function on gpu memory
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights).cuda(GPU_ID)
    else:
        # Initialize SegNet model on cpu memory
        model = SegNet(input_channels=NUM_INPUT_CHANNELS,
                       output_channels=NUM_OUTPUT_CHANNELS)

        # Set target class values
        class_weights = 1.0/val_dataset.get_class_probability()
        # Set loss function
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)


    # Load saved model
    model.load_state_dict(torch.load(SAVED_MODEL_PATH))

    # Start to validation
    validate()


