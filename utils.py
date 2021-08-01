import torch 
import matplotlib.pyplot as plt
import numpy as np 
import cv2

def save_images(out_tensor, input_tensor, filename, epsilon=0.0001):
    orig = input_tensor.detach().numpy().reshape(128,128)
    orig = np.asarray(orig) - epsilon
    orig = orig*255
    gen = out_tensor.detach().numpy().reshape(128,128)
    gen = np.asarray(gen) - epsilon
    gen = gen*255
    pad = np.ones((128,20))*255
    final = np.concatenate((orig.astype(int),pad.astype(int),gen.astype(int)), axis=1)
    cv2.imwrite(filename, final)


