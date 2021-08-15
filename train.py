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

def get_progress_image(test_loader, enc=encoder, dec1=decoder_p1, dec2=decoder_p2, max_images=5):
    '''
    Makes a grid of images for checking progress

    test_loader: data_loader object that contains small number of testing images.
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

    return final_grid

def save_progress_image(test_loader, n_epoch, n_batch, save_dir):
    '''
    saves images to track training progress

    test_loader: data_loader object that contains small number of testing images.
    n_epoch: int, current epoch number.
    n_batch: int, current batch.
    save_dir: str, path to save the progress images.
    '''
    global N_EPOCHS
    _epoch = str(n_epoch).zfill(len(str(N_EPOCHS)))
    _batch = str(n_batch).zfill(8)
    _save_file_name = 'epoch_{_epoch}_batch_{_batch}.jpg' 
    cv2.imwrite(f"{save_dir}/{_save_file_name}", get_progress_image(test_loader))


# Initializing Models and Hyperparams
encoder = Encoder()
decoder_p1 = Decoder()
decoder_p2 = Decoder()
BASE_LR = 0.0003
optim_enc = optim.Adam(encoder.parameters(), lr=BASE_LR)
optim1 = optim.Adam(decoder_p1.parameters(), lr=BASE_LR)
optim2 = optim.Adam(decoder_p2.parameters(), lr=BASE_LR)
