import os

import cv2
import numpy as np
import pandas as pd


def create_dataset(n, path, iscontinue=False, H=256, W=256, R=150):
    if iscontinue:
        i0 = len(os.listdir(path))
        df = pd.read_csv(f"{path}/data.csv")
    else:
        i0 = 0
        df = pd.DataFrame(columns=["name", "x", "y", "r"])
    for i in range(i0, i0 + n):
        img = np.ones((H, W, 3))
        x = np.random.randint(W)
        y = np.random.randint(H)
        r = np.random.randint(1, R)
        c = np.random.randint(255, size=3).astype(int)
        cb = np.random.randint(255, size=3)
        img = (img * cb).astype(np.uint8)
        img = cv2.circle(img, (x, y), r, c.tolist(), -1)
        name = f"{path}/{i}.png"
        cv2.imwrite(name, img)
        df.loc[len(df)] = [f"{i}.png", x, y, r]
    df.to_csv(f"{path}/data.csv")
