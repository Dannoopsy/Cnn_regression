# +
import os

import torch
import torchvision
from Dataset.CreateData import create_dataset
from Dataset.Dataset import CircleDataset
from torch import nn
from torch.utils.data import DataLoader
from trainloop import trainloop

if not os.path.exists("./circles_radxy_train"):
    os.mkdir("./circles_radxy_train")
    create_dataset(50, "./circles_radxy_train")
if not os.path.exists("./circles_radxy_val"):
    os.mkdir("./circles_radxy_val")
    create_dataset(10, "./circles_radxy_val")

dataset = CircleDataset("./circles_radxy_train")
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
valdataset = CircleDataset("./circles_radxy_val")
valloader = DataLoader(valdataset, batch_size=16, shuffle=True)

model = torchvision.models.efficientnet_b0()
model.classifier[-1] = nn.Linear(1280, 3, bias=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.MSELoss(reduction="sum")
l1 = nn.L1Loss(reduction="sum")

epochs = 2

model = trainloop(model, optimizer, device, l1, dataloader, valloader, epochs)

torch.save(model.state_dict(), "./efnetb0.pkl")
