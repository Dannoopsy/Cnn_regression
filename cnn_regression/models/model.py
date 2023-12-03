import torch
import torchvision
from torch import nn


class RegModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.efficientnet_b0()
        self.model.classifier[-1] = nn.Linear(1280, 1, bias=True)

    def forward(self, x):
        return self.model(x)
