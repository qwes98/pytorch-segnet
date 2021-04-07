"""
Train a SegNet model


Usage:
python train.py --data_root /home/SharedData/intern_sayan/PascalVOC2012/data/VOCdevkit/VOC2012/ \
                --train_path ImageSets/Segmentation/train.txt \
                --img_dir JPEGImages \
                --mask_dir SegmentationClass \
                --save_dir /home/SharedData/intern_sayan/PascalVOC2012/ \
                --checkpoint /home/SharedData/intern_sayan/PascalVOC2012/model_best.pth \
                --gpu 1
"""

from __future__ import print_function

# Import modules
import argparse         # module to parse argument value
from dataset import PascalVOCDataset, NUM_CLASSES   # module to manage dataset
from model import SegNet    # model module
import os
import time

# Import torch modules
import torch
from torch.utils.data import DataLoader


# Constants
NUM_INPUT_CHANNELS = 3              # Set input image channel
NUM_OUTPUT_CHANNELS = NUM_CLASSES   # Set output channel (same as # of output class)

NUM_EPOCHS = 6000   # Number of epoch (how many we will use dataset to train model)

LEARNING_RATE = 1e-6    # Learning rate
MOMENTUM = 0.9          # Momentum hyperparameter (used in optimizer)
BATCH_SIZE = 16         # Mini batch size


# Arguments
parser = argparse.ArgumentParser(description='Train a SegNet model')

# Path to directory which contains data
parser.add_argument('--data_root', required=True)
# Path to file contains train data path
parser.add_argument('--train_path', required=True)
# Path to image directory name
parser.add_argument('--img_dir', required=True)
# Path to target directory name
parser.add_argument('--mask_dir', required=True)
# Path to output saved directory
parser.add_argument('--save_dir', required=True)
# Path to pretrained(checkpoint) weight file
parser.add_argument('--checkpoint')
# Set whether to use gpu
parser.add_argument('--gpu', type=int)

# Parse arguments
args = parser.parse_args()

# Function which contains traning algorithm
def train():
    # Flag to check better model than before
    is_better = True
    # Store previous loss
    prev_loss = float('inf')

    # Change model to train mode
    model.train()

    # Iterate training through dataset epoch time
    for epoch in range(NUM_EPOCHS):
        # Initialiize loss value
        loss_f = 0
        # Store time before start training
        t_start = time.time()

        # Iterate mini-batch
        for batch in train_dataloader:
            # Make input variable by data
            input_tensor = torch.autograd.Variable(batch['image'])
            # Make target variable by data
            target_tensor = torch.autograd.Variable(batch['mask'])

            # If could use cuda
            if CUDA:
                # Store input tensor on gpu memory
                input_tensor = input_tensor.cuda(GPU_ID)
                # Store target tensor on gpu memory
                target_tensor = target_tensor.cuda(GPU_ID)

            # Inference
            predicted_tensor, softmaxed_tensor = model(input_tensor)

            # Initialize gradient value to zero to calculate gradient
            optimizer.zero_grad()
            # Calculate loss
            loss = criterion(predicted_tensor, target_tensor)
            # Calculate all gradients to backward
            loss.backward()
            # Update weights by gredients
            optimizer.step()

            # Sum loss value
            loss_f += loss.float()
            # Store prediction
            prediction_f = softmaxed_tensor.float()

        # Calculate one epoch timej
        delta = time.time() - t_start
        # Check this epoch makes model better
        is_better = loss_f < prev_loss

        # If this model is better than before
        if is_better:
            # Store loss value
            prev_loss = loss_f
            # Save this model to Best
            torch.save(model.state_dict(), os.path.join(args.save_dir, "model_best.pth"))

        # Print this epoch result
        print("Epoch #{}\tLoss: {:.8f}\t Time: {:2f}s".format(epoch+1, loss_f, delta))


if __name__ == "__main__":
    # Read data root path
    data_root = args.data_root
    # Set train data full path
    train_path = os.path.join(data_root, args.train_path)
    # Set image directory full path
    img_dir = os.path.join(data_root, args.img_dir)
    # Set target directory full path
    mask_dir = os.path.join(data_root, args.mask_dir)

    # Set to use cuda
    CUDA = args.gpu is not None
    # Set gpu id
    GPU_ID = args.gpu

    # Initialize training dataset
    train_dataset = PascalVOCDataset(list_file=train_path,
                                     img_dir=img_dir,
                                     mask_dir=mask_dir)

    # Load dataset by dataloader
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=4)

    # If we use cuda
    if CUDA:
        # Initialize model and load on gpu memory
        model = SegNet(input_channels=NUM_INPUT_CHANNELS,
                       output_channels=NUM_OUTPUT_CHANNELS).cuda(GPU_ID)

        # Get class probability and load on gpu memory
        class_weights = 1.0/train_dataset.get_class_probability().cuda(GPU_ID)
        # Set loss criterion
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights).cuda(GPU_ID)
    # If we use cpu
    else:
        # Initialize model
        model = SegNet(input_channels=NUM_INPUT_CHANNELS,
                       output_channels=NUM_OUTPUT_CHANNELS)

        # Get class probability
        class_weights = 1.0/train_dataset.get_class_probability()
        # Set loss creterion
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # If we use checkpoint
    if args.checkpoint:
        # Load checkpoint weight
        model.load_state_dict(torch.load(args.checkpoint))

    # Initialize optimizer (use adam)
    optimizer = torch.optim.Adam(model.parameters(),
                                     lr=LEARNING_RATE)

    # Start to training
    train()
