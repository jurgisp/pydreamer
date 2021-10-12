from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from models.functions import *

# TODO: this is work-in-progress attempt to use type aliases to indicate the shapes of tensors.
# N = 50         (TBTT length)
# B = 50         (batch size)
# A = 3          (action dim)
# I = 1/3/10     (IWAE)
# F = 2048+32    (feature_dim)
# H = 10         (dream horizon)
# J = H+1 = 11
# M = N*B*I = 2500
TensorNBCHW = Tensor
TensorNB = Tensor
TensorNBICHW = Tensor
TensorNBI4 = Tensor
TensorJMF = Tensor
TensorJM2 = Tensor
TensorHMA = Tensor
TensorHM = Tensor
TensorJM = Tensor

IntTensorNBIHW = Tensor
StateB = Tuple[Tensor, Tensor]
StateNB = Tuple[Tensor, Tensor]


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


class CategoricalSupport(D.Categorical):

    def __init__(self, logits, support):
        assert logits.shape[-1:] == support.shape
        super().__init__(logits=logits)
        self._support = support

    @property
    def mean(self):
        return torch.einsum('...i,i->...', self.probs, self._support)


class TrainableModel(nn.Module):

    @property
    @abstractmethod
    def submodels(self) -> Tuple:
        ...

    @abstractmethod
    def optimizers(self, conf) -> Tuple:
        ...

    @abstractmethod
    def grad_clip(self, conf) -> Dict:
        ...

    @abstractmethod
    def init_state(self, batch_size: int) -> Any:
        ...

    @abstractmethod
    def forward(self,
                image: TensorNBCHW,   # (1,B,C,H,W)
                vecobs: Tensor,       # (1,V)
                prev_reward: Tensor,  # (1,B)
                prev_action: Tensor,  # (1,B,A)
                reset: Tensor,        # (1,B)
                in_state: Any,
                ) -> Tuple:
        ...

    @abstractmethod
    def train(self,
              image: TensorNBCHW,
              vecobs: Tensor,
              reward: Tensor,
              terminal: Tensor,
              action_prev: Tensor,
              reset: Tensor,
              map: Tensor,
              map_coord: Tensor,
              map_seen_mask: Tensor,
              in_state: Any,
              I: int = 1,
              H: int = 1,
              imagine_dropout=0,
              do_image_pred=False,
              do_output_tensors=False,
              do_dream_tensors=False,
              ) -> Tuple:
        ...
