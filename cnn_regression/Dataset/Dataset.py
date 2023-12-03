import os

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms as T


class AgeDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=T.ToTensor()):
        self.root = root
        self.transforms = transforms
        self.files = os.listdir(root)

    def __getitem__(self, idx):
        file = self.files[idx]
        img_path = os.path.join(self.root, file)
        image = Image.open(img_path).convert("RGB")
        image = self.transforms(image)
        age = file[:file.find('_')]
        return image, torch.tensor(age)

    def __len__(self):
        return len(self.files)
