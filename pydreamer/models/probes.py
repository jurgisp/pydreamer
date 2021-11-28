from typing import Any, Tuple

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

from .a2c import *
from .common import *
from .functions import *
from .decoders import *
from .rnn import *
from .rssm import *


class MapProbeHead(nn.Module):

    def __init__(self, map_state_dim, conf):
        super().__init__()
        if conf.map_decoder == 'dense':
            self.decoder = CatImageDecoder(in_dim=map_state_dim,
                                             out_shape=(conf.map_channels, conf.map_size, conf.map_size),
                                             hidden_dim=conf.map_hidden_dim,
                                             hidden_layers=conf.map_hidden_layers,
                                             layer_norm=conf.layer_norm)
        else:
            raise NotImplementedError(conf.map_decoder)
            # self.decoder = ConvDecoder(in_dim=map_state_dim,
            #                            mlp_layers=2,
            #                            layer_norm=conf.layer_norm,
            #                            out_channels=conf.map_channels)

    def training_step(self,
                      features: TensorTBIF,
                      obs: Dict[str, Tensor],
                      ):
        I = features.shape[2]
        map_coord = insert_dim(obs['map_coord'], 2, I)
        map_features = torch.cat((features, map_coord), dim=-1)

        _, loss, map_pred = self.decoder.training_step(map_features, obs['map'])

        with torch.no_grad():
            map_pred = map_pred.detach()
            acc_map = self.accuracy(map_pred, obs['map'])
            acc_map_seen = self.accuracy(map_pred, obs['map'], obs['map_seen_mask'])
            tensors = dict(map_rec=map_pred,
                           loss_map=loss.detach(),
                           acc_map=acc_map)
            metrics = dict(loss_map=loss.mean(),
                           acc_map=nanmean(acc_map),
                           acc_map_seen=nanmean(acc_map_seen))

        return loss.mean(), metrics, tensors

    def accuracy(self, output: TensorTBCHW, target: Union[TensorTBCHW, IntTensorTBHW], map_seen_mask: Optional[Tensor] = None):
        if len(output.shape) == len(target.shape):
            target = target.argmax(dim=-3)  # float(*,C,H,W) => int(*,H,W)
        output, bd = flatten_batch(output, 3)
        target, _ = flatten_batch(target, 2)

        acc = output.argmax(dim=-3) == target
        if map_seen_mask is None:
            acc = acc.to(torch.float).mean([-1, -2])
        else:
            map_seen_mask, _ = flatten_batch(map_seen_mask, 2)  # (*,H,W)
            acc = (acc * map_seen_mask).sum([-1, -2]) / map_seen_mask.sum([-1, -2])
        acc = unflatten_batch(acc, bd)  # (T,B)
        return acc


class NoProbeHead(nn.Module):

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=True)

    def training_step(self,
                      features: TensorTBIF,
                      obs: Dict[str, Tensor],
                      ):
        return torch.square(self.dummy), {}, {}
