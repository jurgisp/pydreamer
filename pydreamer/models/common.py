from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from .functions import *

# This is a work-in-progress attempt to use type aliases to indicate the shapes of tensors.
# T = 50         (TBTT length)
# B = 50         (batch size)
# I = 1/3/10     (IWAE samples)
# A = 3          (action dim)
# E              (embedding dim)
# F = 2048+32    (feature_dim)
# H = 10         (dream horizon)
# J = H+1 = 11
# M = T*B*I = 2500
TensorTBCHW = Tensor
TensorTB = Tensor
TensorTBE = Tensor
TensorTBICHW = Tensor
TensorTBIF = Tensor
TensorTBI = Tensor
TensorJMF = Tensor
TensorJM2 = Tensor
TensorHMA = Tensor
TensorHM = Tensor
TensorJM = Tensor

IntTensorTBHW = Tensor
StateB = Tuple[Tensor, Tensor]
StateTB = Tuple[Tensor, Tensor]


class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim, hidden_layers, layer_norm, activation=nn.ELU):
        super().__init__()
        self.out_dim = out_dim
        norm = nn.LayerNorm if layer_norm else NoNorm
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
        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x, bd = flatten_batch(x)
        y = self.model(x)
        y = unflatten_batch(y, bd)
        return y


class NoNorm(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x


class CategoricalSupport(D.Categorical):

    def __init__(self, logits, support):
        assert logits.shape[-1:] == support.shape
        super().__init__(logits=logits)
        self.support = support

    @property
    def mean(self):
        return torch.einsum('...i,i->...', self.probs, self.support)
