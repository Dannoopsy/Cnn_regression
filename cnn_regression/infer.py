# +
import os

import pandas as pd
import torch
import torchvision
from Dataset.CreateData import create_dataset
from Dataset.Dataset import CircleDataset
from torch import nn
from torch.utils.data import DataLoader
from utils.utils import testloop

if __name__ == "__main__":
    model = torchvision.models.efficientnet_b0()
    model.classifier[-1] = nn.Linear(1280, 3, bias=True)
    model.load_state_dict(torch.load("./efnetb0.pkl"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if not os.path.exists("./circles_radxy_val"):
        os.mkdir("./circles_radxy_val")
        create_dataset(50, "./circles_radxy_val")
    valdataset = CircleDataset("./circles_radxy_val")
    valloader = DataLoader(valdataset, batch_size=16, shuffle=True)

    metrics = {"l1": nn.L1Loss(reduction="mean"), "mse": nn.MSELoss(reduction="mean")}

    metric_values = testloop(model, device, metrics, valloader)
    print(metric_values)
    res = pd.DataFrame(metric_values, index=["efnetb0.pkl"])
    res.to_csv("./res.csv")
