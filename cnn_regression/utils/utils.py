# +
import pandas as pd
import torch

import os

import torch
from Dataset.CreateData import create_dataset
from Dataset.Dataset import CircleDataset
from torch import nn
from torch.utils.data import DataLoader
from utils.utils import trainloop
from models.model import RegModel


# -

def trainloop(model, optimizer, device, criterion, dataloader, valloader, epochs=10):
    for i in range(epochs):
        model.train()
        lt = 0
        for x, y in dataloader:
            out = model(x.to(device))
            loss = criterion(y.to(device).float(), out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lt += loss.item()

        lv = 0
        model.eval()

        with torch.no_grad():
            for x, y in valloader:
                out = model(x.to(device))
                loss = criterion(y.to(device).float(), out)
                lv += loss.item()

        print(
            f"epoch = {i}, train loss = {lt / len(dataloader)}, val loss = {lv / len(valloader)}"
        )
    model.cpu()
    return model


def testloop(model, device, metrics, loader):
    metric_values = {m: 0 for m in metrics}
    df = pd.DataFrame(columns=["name", "x", "y", "r"])
    i = 0
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device))
            for an in out:
                xa, ya, ra = an.tolist()
                df.loc[len(df)] = [f"{i}.png", xa, ya, ra]
                i += 1
            for m in metrics:
                metric_values[m] += metrics[m](y.to(device), out)

    for m in metrics:
        metric_values[m] = metric_values[m].item() / len(loader)
    df.to_csv("./res.csv")
    return metric_values


def train():
    dataset = AgeDataset("./circles_radxy_train")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    valdataset = CircleDataset("./circles_radxy_val")
    valloader = DataLoader(valdataset, batch_size=16, shuffle=True)

    model = RegModel()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    mse = nn.MSELoss(reduction="mean")
    l1 = nn.L1Loss(reduction="mean")
    epochs = 3

    model = trainloop(model, optimizer, device, mse, dataloader, valloader, epochs)
    torch.save(model.state_dict(), "./efnetb0.pkl")
