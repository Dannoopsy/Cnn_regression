import pandas as pd
import torch


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
