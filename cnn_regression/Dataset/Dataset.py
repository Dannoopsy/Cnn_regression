import os

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms as T

default_transform = T.ToTensor()


class CircleDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=default_transform):
        self.root = root
        self.transforms = transforms
        self.df = pd.read_csv(f"{root}/data.csv")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root, row["name"])
        image = Image.open(img_path).convert("RGB")
        image = self.transforms(image)
        r = row["r"]
        x = row["x"]
        y = row["y"]
        return image, torch.tensor([r, x, y])

    def __len__(self):
        return len(self.df.index)
