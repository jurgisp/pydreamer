import torch
import torch.nn as nn
from torch import Tensor

from modules_tools import *


class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim, hidden_layers, layer_norm, activation=nn.ELU):
        norm = nn.LayerNorm if layer_norm else NoNorm
        super().__init__()
        layers = []
        for i in range(hidden_layers):
            layers += [
                nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim),
                norm(hidden_dim, eps=1e-3),
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


class NoNorm(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x
