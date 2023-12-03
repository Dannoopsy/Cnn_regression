import os

import torch
from PIL import Image

class AgeDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.files = os.listdir(root)

    def __getitem__(self, idx):
        file = self.files[idx]
        img_path = os.path.join(self.root, file)
        image = Image.open(img_path).convert("RGB")
        image = self.transforms(image)
        age = float(file[: file.find('_')])
        return image, torch.tensor(age)

    def __len__(self):
        return len(self.files)
