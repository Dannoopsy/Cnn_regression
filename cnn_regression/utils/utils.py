import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm

from cnn_regression.dataset.dataset import AgeDataset
from cnn_regression.models.model import RegModel


def trainloop(model, optimizer, device, criterion, dataloader, valloader, epochs=10):
    for i in range(epochs):
        model.train()
        lt = 0
        for x, y in tqdm(dataloader):
            out = model(x.to(device))
            loss = criterion(y.to(device).float().view(out.shape), out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lt += loss.item()

        lv = 0
        model.eval()

        with torch.no_grad():
            for x, y in tqdm(valloader):
                out = model(x.to(device))
                loss = criterion(y.to(device).float().view(out.shape), out)
                lv += loss.item()

        print(
            f"epoch = {i}, train loss = {lt / len(dataloader)}, val loss = {lv / len(valloader)}"
        )
    model.cpu()
    return model


def testloop(model, device, metrics, loader, dataset, path_to_res_file):
    metric_values = {m: 0 for m in metrics}
    df = pd.DataFrame(columns=["name", "age"])
    i = 0
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device))
            for an in out.squeeze():
                age = int(an)
                df.loc[len(df)] = [dataset.files[i], age]
                i += 1
            for m in metrics:
                metric_values[m] += metrics[m](y.to(device).view(out.shape), out)

    for m in metrics:
        metric_values[m] = metric_values[m].item() / len(loader)
    df.to_csv(path_to_res_file)
    return metric_values


def train(cfg):
    train_path = cfg['data']['train_path']
    val_path = cfg['data']['val_path']
    epochs = cfg['training']['epochs']
    out_model_path = cfg['data']['out_model_path']
    batch_size = cfg['training']['batch_size']
    max_data = cfg['training']['max_data']
    optimizer = cfg['training']['optimizer']
    lr = cfg['training']['learning_rate']
    loss = cfg['training']['loss']
    dataset = AgeDataset(train_path, T.ToTensor())
    indices = (
        list(set(range(max_data)) & set(range(len(dataset))))
        if max_data > 0
        else list(range(len(dataset)))
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=torch.utils.data.SubsetRandomSampler(indices, generator=None),
    )
    valdataset = AgeDataset(val_path, T.ToTensor())
    valloader = DataLoader(valdataset, batch_size=batch_size, shuffle=True)
    model = RegModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    else:
        print('не знаю такого, использую SGD')
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    if loss == 'mse':
        loss = nn.MSELoss(reduction="mean")
    else:
        loss = nn.L1Loss(reduction="mean")
    model = trainloop(model, optimizer, device, loss, dataloader, valloader, epochs)
    torch.save(model.state_dict(), out_model_path)


def infer(
    path_to_data='./crop_part1/val/',
    path_to_model="./efnetb0.pkl",
    path_to_res_file="./res.csv",
):
    model = RegModel()
    model.load_state_dict(torch.load(path_to_model))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    valdataset = AgeDataset(path_to_data, T.ToTensor())
    valloader = DataLoader(valdataset, batch_size=16, shuffle=False)
    metrics = {"l1": nn.L1Loss(reduction="mean"), "mse": nn.MSELoss(reduction="mean")}
    metric_values = testloop(
        model, device, metrics, valloader, valdataset, path_to_res_file
    )
    print(metric_values)
