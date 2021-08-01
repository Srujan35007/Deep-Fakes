import torch
import numpy as np 
import cv2 
import os 

class FaceData(torch.utils.data.Dataset):
    def __init__(self, images_dir, transform=None):
        self.file_paths = []
        for roots,dirs,files in os.walk(images_dir):
            for file_ in files:
                if file_.endswith('.jpg'):
                    self.file_paths.append(f'{images_dir}/{file_}')
        self.file_paths.sort()
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        img = cv2.imread(self.file_paths[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.asarray(cv2.resize(img, (128,128)))/255. + 0.0001
        if self.transform:
            img = self.transform(img)
        return img.float().view(1,128,128) 
