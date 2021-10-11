from typing import Callable, Dict, List, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch import Tensor, Size

import models.rnn as my


def flatten(x: Tensor) -> Tensor:
    # (N, B, ...) => (N*B, ...)
    return torch.reshape(x, (-1,) + x.shape[2:])


def unflatten(x: Tensor, n: int) -> Tensor:
    # (N*B, ...) => (N, B, ...)
    return torch.reshape(x, (n, -1) + x.shape[1:])


def flatten_batch(x: Tensor, nonbatch_dims=1) -> Tuple[Tensor, Size]:
    # (b1,b2,..., X) => (B, X)
    if nonbatch_dims > 0:
        batch_dim = x.shape[:-nonbatch_dims]
        x = torch.reshape(x, (-1,) + x.shape[-nonbatch_dims:])
        return x, batch_dim
    else:
        batch_dim = x.shape
        x = torch.reshape(x, (-1,))
        return x, batch_dim


def unflatten_batch(x: Tensor, batch_dim: Union[Size, Tuple]) -> Tensor:
    # (B, X) => (b1,b2,..., X)
    x = torch.reshape(x, batch_dim + x.shape[1:])
    return x


def diag_normal(x: Tensor, min_std=0.1, max_std=2.0):
    # DreamerV2:
    # std = {
    #     'softplus': lambda: tf.nn.softplus(std),
    #     'sigmoid2': lambda: 2 * tf.nn.sigmoid(std / 2),
    # }[self._std_act]()
    # std = std + self._min_std

    mean, std = x.chunk(2, -1)
    # std = F.softplus(std) + min_std
    std = max_std * torch.sigmoid(std) + min_std
    return D.independent.Independent(D.normal.Normal(mean, std), 1)


def rsample(x: Tensor, noise: Tensor, min_std=0.1, max_std=2.0):
    mean, std = x.chunk(2, -1)
    std = max_std * torch.sigmoid(std) + min_std
    return mean + noise * std


def init_weights_tf2(m):
    # Match TF2 initializations
    if type(m) in {nn.Conv2d, nn.ConvTranspose2d, nn.Linear}:
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)
    if type(m) == nn.GRUCell or type(m) == my.GRUCell:
        nn.init.xavier_uniform_(m.weight_ih.data)
        nn.init.orthogonal_(m.weight_hh.data)
        nn.init.zeros_(m.bias_ih.data)
        nn.init.zeros_(m.bias_hh.data)
    if type(m) == my.NormGRUCell or type(m) == my.NormGRUCellLateReset:
        nn.init.xavier_uniform_(m.weight_ih.weight.data)
        nn.init.orthogonal_(m.weight_hh.weight.data)


def logavgexp(x: Tensor, dim: int) -> Tensor:
    if x.size(dim) > 1:
        # TODO: cast to float32 here for IWAE?
        return x.logsumexp(dim=dim) - np.log(x.size(dim))
    else:
        return x.squeeze(dim)


def map_structure(data: Tuple[Tensor, ...], f: Callable[[Tensor], Tensor]) -> Tuple[Tensor, ...]:
    # Like tf.nest.map_structure
    assert isinstance(data, tuple), 'Not implemented for other types'
    return tuple(f(d) for d in data)


def stack_structure(data: List[Tuple[Tensor, ...]]) -> Tuple[Tensor, ...]:
    assert isinstance(data[0], tuple), 'Not implemented for other types'
    n = len(data[0])
    return tuple(
        torch.stack([d[i] for d in data])
        for i in range(n)
    )


def cat_structure_np(datas: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    assert isinstance(datas[0], dict), 'Not implemented for other types'
    keys = set(datas[0].keys())
    for d in datas[1:]:
        keys.intersection_update(d.keys())
    return {  # type: ignore
        k: np.concatenate([d[k] for d in datas])
        for k in keys
    }


def stack_structure_np(datas: Tuple[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    assert isinstance(datas[0], dict), 'Not implemented for other types'
    keys = set(datas[0].keys())
    for d in datas[1:]:
        keys.intersection_update(d.keys())
    return {  # type: ignore
        key: np.stack([d[key] for d in datas])
        for key in keys
    }


def map_structure_np(data: Dict[str, np.ndarray], f: Callable[[np.ndarray], np.ndarray]) -> Dict[str, np.ndarray]:
    assert isinstance(data, dict), 'Not implemented for other types'
    return {k: f(v) for k, v in data.items()}


def nanmean(x: Tensor) -> Tensor:
    return torch.nansum(x) / (~torch.isnan(x)).sum()