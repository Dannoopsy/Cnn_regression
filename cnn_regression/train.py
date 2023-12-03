import os
import torch
from Dataset.Dataset import AgeDataset
from torch import nn
from torch.utils.data import DataLoader
from utils.utils import train
from models.model import RegModel
import fire


if __name__ == "__main__":
    fire.Fire()