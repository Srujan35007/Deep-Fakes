from glob import glob 
import random
import torch 
import cv2

class FaceData(torch.utils.data.Dataset):
    def __init__(self, images_path, num_images, transform):
        image_paths = glob(f"{images_path}/*.jpg")
        random.shuffle(image_paths)
        n_paths = min(num_images, len(image_paths))
        self.image_paths = image_paths[:n_paths]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        _img = cv2.imread(self.image_paths[index])
        _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
        if self.transform:
            _img = self.transform(_img)
        return _img
