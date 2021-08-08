import torch
import torch.nn as nn
from torch import Tensor

from modules_tools import *


class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim, hidden_layers, activation=nn.ELU):
        super().__init__()
        layers = []
        for i in range(hidden_layers):
            layers += [
                nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                activation()
            ]
        layers += [
            nn.Linear(hidden_dim, out_dim),
        ]
        if out_dim == 1:
            layers += [
                nn.Flatten(0),
            ]   
        self._model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x, bd = flatten_batch(x)
        y = self._model(x)
        y = unflatten_batch(y, bd)
        return y
