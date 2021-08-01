import os 
import time 
import torch 
import torch.nn as nn 
from torchvision import transforms
import numpy as np 
import cv2 
import matplotlib.pyplot as plt 
from models import AutoEncoderConv
from dataset import FaceData
print(f"Imports complete")

def plot_image(out_tensor):
    img = out_tensor.view(128,128).detach().numpy().reshape(128,128) - 0.0001
    img = (img*255).astype(int)
    plt.imshow(img,cmap='gray')
    plt.show()

test_data = FaceData('./Data/Obama_Faces/val', transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1,shuffle=True)

Obama = AutoEncoderConv(1)
Biden = AutoEncoderConv(1)
Obama = torch.load('./Trained_Models/Obama_AutoEncoderConv_Face_2021_07_30_17_09_52_valLoss0.016778_valAcc88.56.pt')
Biden = torch.load('./Trained_Models/Biden_AutoEncoderConv_Face_2021_07_30_21_32_59_valLoss0.012363_valAcc91.47.pt')
for idx, data in enumerate(test_loader):
    out = Biden.Decoder(Biden.Encoder(data.view(1,1,128,128)))
    plot_image(out)
    if idx > 6:
        break
