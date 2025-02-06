import torch 
import os 
import yaml
from torch.utils.data import DataLoader, Dataset
import numpy as np
import albumentations as A
from PIL import Image
from utils.helper import *

class FaceCycleGANDataset(Dataset):
    def __init__(self, root_face, root_ukiyo, transform=None):
        super().__init__()
        self.root_face = root_face
        self.root_ukiyo = root_ukiyo
        self.transform = transform

        self.face_images = os.listdir(root_face)
        self.ukiyo_images = os.listdir(root_ukiyo)
        self.len = max(len(self.face_images), len(self.ukiyo_images))

        self.face_len = len(self.face_images)
        self.ukiyo_len = len(self.ukiyo_images)
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        face_img = self.face_images[idx % self.face_len]
        ukiyo_img = self.ukiyo_images[idx % self.ukiyo_len]

        face_path = os.path.join(self.root_face, face_img)
        ukiyo_path = os.path.join(self.root_ukiyo, ukiyo_img)

        face_img = np.array(Image.open(face_path).convert('RGB'))
        ukiyo_img = np.array(Image.open(ukiyo_path).convert('RGB'))

        if self.transform:
            augmentations = self.transform(image=face_img, image0=ukiyo_img)
            face_img = augmentations['image']
            ukiyo_img = augmentations['image0']

        return face_img, ukiyo_img
