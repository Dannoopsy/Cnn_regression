# +
from typing import NamedTuple
from PIL import Image

import os
import torch
from torch import nn
from torchvision import transforms as T
import torchvision
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.functional import mse_loss
from torch import optim
from torchvision.transforms.functional import crop
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets
import pandas as pd
from Dataset.Dataset import CircleDataset
from trainloop import trainloop
from tqdm import tqdm
from Dataset.CreateData import create_dataset

if not os.path.exists('./circles_radxy_train'):
    os.mkdir('./circles_radxy_train')
    create_dataset(50, './circles_radxy_train')
if not os.path.exists('./circles_radxy_val'):
    os.mkdir('./circles_radxy_val')
    create_dataset(10, './circles_radxy_val')

dataset = CircleDataset('./circles_radxy_train')
dataloader = DataLoader(dataset, batch_size = 16, shuffle=True)
valdataset = CircleDataset('./circles_radxy_val')
valloader = DataLoader(valdataset, batch_size = 16, shuffle=True)

model = torchvision.models.efficientnet_b0()
model.classifier[-1] = nn.Linear(1280, 3, bias=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.MSELoss(reduction='sum')
l1 = nn.L1Loss(reduction='sum')

epochs = 2

model = trainloop(model, optimizer, device, l1, dataloader, valloader, epochs)

torch.save(model.state_dict(), './efnetb0.pkl')
