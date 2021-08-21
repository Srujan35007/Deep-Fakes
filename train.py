#!/bin/python3
'''
Author: Suddala Srujan
Train script for training Encoder-Decoder networks for deep-fakes.
[] Implement Encoder Decoder training loops
[] Implement conditions to Autosave the models
[] Implement saving generated images to track progress
'''

import os 
import time 
import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import numpy as np 
import cv2
import matplotlib.pyplot as plt 
from argparse import ArgumentParser as AP 
from datetime import datetime 
from models import Encoder, Decoder
from dataset import FaceData
print(f"Imports complete\n")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cpu = torch.device('cpu')
IMG_SIZE = 128
N_CHANNELS = 3
BATCH_SIZE = 8 # Training batch size

# Argument parsing
parser = AP()
parser.add_argument('-s', '--autosave', choices=[0,1], default=0,
                    type=int, help='Save the model when there is a new local val_loss minima?')
parser.add_argument('-n', '--n_epochs', default=5,
                    type=int, help='No. of epochs to train the model(s)')
args = parser.parse_args()
N_EPOCHS = args.n_epochs
AUTOSAVE = args.autosave


# Helper functions
get_utc_timestamp = lambda : datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

def get_progress_image(test_loaders, enc=encoder, dec1=decoder_p1, dec2=decoder_p2, max_images=5):
    '''
    Makes a grid of images for checking progress

    test_loaders: list, data_loader objs that contain small number of testing images of p1 and p2.
    enc: The Encoder network that is shared between person 1 and person 2.
    dec1: The Decoder network of person 1.
    dec2: The Decoder network of person 2.
    max_images: int, Maximum number of images that should be in the progress images grid.
    returns a numpy array of the final image.
    '''
    global device
    global cpu 
    # TODO: torchvision.utils - make grid
    # 1. dec1(enc(x1))
    # 2. dec2(enc(x2))
    # 3. dec2(enc(x1))
    # 4. dec1(enc(x2))
    enc.eval()
    dec1.eval()
    dec2.eval()
    grid = []
    for idx, p1_img, p2_img in enumerate(zip(*test_loaders)):
        grid.append(dec1(enc(p1_img.to(device))).cpu())
        grid.append(dec2(enc(p2_img.to(device))).cpu())
        grid.append(dec1(enc(p2_img.to(device))).cpu())
        grid.append(dec2(enc(p1_img.to(device))).cpu())
        
    return final_grid

def save_progress_image(test_loaders, n_epoch, n_batch, save_dir):
    '''
    saves images to track training progress

    test_loaders: list, data_loader objs that contain small number of testing images of p1 and p2.
    n_epoch: int, current epoch number.
    n_batch: int, current batch.
    save_dir: str, path to save the progress images.
    '''
    global N_EPOCHS
    _epoch = str(n_epoch).zfill(len(str(N_EPOCHS)))
    _batch = str(n_batch).zfill(8)
    _save_file_name = 'epoch_{_epoch}_batch_{_batch}.jpg' 
    cv2.imwrite(f"{save_dir}/{_save_file_name}", get_progress_image(test_loaders))


# Get data
composed_transforms = transforms.ToTensor()
train_loader_p1 = DataLoader(FaceData('./Datasets/Obama/train', transform=composed_transforms),
                            batch_size=BATCH_SIZE, shuffle=True)
train_loader_p2 = DataLoader(FaceData('./Datasets/Biden/train', transform=composed_transforms),
                            batch_size=BATCH_SIZE, shuffle=True)
val_loader_p1 = DataLoader(FaceData('./Datasets/Obama/val', transform=composed_transforms))
val_loader_p2 = DataLoader(FaceData('./Datasets/Obama/val', transform=composed_transforms))
test_loader_p1 = DataLoader(FaceData('./Datasets/Obama/test', transform=composed_transforms))
test_loader_p2 = DataLoader(FaceData('./Datasets/Obama/test', transform=composed_transforms))


# Initializing Models and Hyperparams
encoder = Encoder()
decoder_p1 = Decoder()
decoder_p2 = Decoder()
BASE_LR = 0.0003
optim_enc = optim.Adam(encoder.parameters(), lr=BASE_LR)
optim1 = optim.Adam(decoder_p1.parameters(), lr=BASE_LR)
optim2 = optim.Adam(decoder_p2.parameters(), lr=BASE_LR)


