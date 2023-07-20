import os
import random as rn
import numpy as np
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision import transforms
from torchvision.transforms import ToTensor

import cv2
from glob import glob

import matplotlib.pyplot as plt
from skimage.transform import rotate, AffineTransform, warp
import albumentations as A
import albumentations.augmentations.functional as F
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
from torch.utils.data import Dataset, DataLoader


class ImageTransform():
    def __init__(self, resize, mean, std):
        self.data_transform = {
            "train": transforms.Compose([
                transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]), 
            "val": transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }
    
    def __call__(self, image, phase="train"):
        return self.data_transform[phase](image)
    
class Load_Data(Dataset):
    def __init__(self, folder_path, transform=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fpaths = glob(folder_path + "/images/*.png")
        self.transform = transform

    def __getitem__(self, idx):
        path = self.fpaths[idx]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = np.asarray(Image.open(path))
        mask = cv2.imread(path.replace("images", "masks"))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        # mask = np.asarray(Image.open(path.replace("images", "masks")))
        if self.transform is not None:
            transformed = self.transform(image=img, mask=mask)
            img_t = transformed["image"]
            mask = transformed["mask"]
        img_t = torch.tensor(img_t)
        mask = torch.tensor(mask)
        img = cv2.resize(img, (256, 256))
        return [img_t, mask]

    def __len__(self):
        return len(self.fpaths)
