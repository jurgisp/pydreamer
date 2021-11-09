from typing import Any, Tuple

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

from models.a2c import *
from models.common import *
from models.functions import *
from models.io import *
from models.rnn import *
from models.rssm import *


class MapProbeHead(nn.Module):

    def __init__(self, map_state_dim, conf):
        super().__init__()
        if conf.map_decoder == 'dense':
            self.decoder = DenseDecoder(in_dim=map_state_dim,
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
                      features: TensorNBIF,
                      obs: Dict[str, Tensor],
                      ):
        I = features.shape[2]
        map_coord = insert_dim(obs['map_coord'], 2, I)  # TODO: find a consistent solution where to put all these expand-I
        map = insert_dim(obs['map'], 2, I)
        map_seen_mask = obs['map_seen_mask']

        map_features = torch.cat((features, map_coord), dim=-1)
        map_pred = self.decoder.forward(map_features)
        loss = self.decoder.loss(map_pred, map)  # (N,B,I)
        loss = -logavgexp(-loss, dim=-1)  # (N,B,I) => (N,B)

        with torch.no_grad():
            acc_map = self.decoder.accuracy(map_pred, map)
            acc_map_seen = self.decoder.accuracy(map_pred, map, map_seen_mask)
            tensors = dict(loss_map=loss.detach(),
                           acc_map=acc_map)
            metrics = dict(loss_map=loss.mean(),
                           acc_map=nanmean(acc_map),
                           acc_map_seen=nanmean(acc_map_seen))
            tensors['map_rec'] = self.decoder.to_distr(map_pred)  # TODO: output from decoder.training_step()

        return loss.mean(), metrics, tensors


class NoProbeHead(nn.Module):

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=True)

    def training_step(self,
                      features: TensorNBIF,
                      obs: Dict[str, Tensor],
                      ):
        return torch.square(self.dummy), {}, {}
