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

class MapGoalsProbe(nn.Module):
    """Combined MapProbeHead and GoalsProbe."""

    def __init__(self, state_dim, conf):
        super().__init__()
        self.map_probe = MapProbeHead(state_dim + 4, conf)
        self.goals_probe = GoalsProbe(state_dim, conf)

    def training_step(self, features: TensorTBIF, obs: Dict[str, Tensor]):
        loss_map, metrics_map, tensors_map = self.map_probe.training_step(features, obs)
        loss_goals, metrics_goals, tensors_goals = self.goals_probe.training_step(features, obs)
        loss_total = loss_map + loss_goals
        metrics = dict(**metrics_map, **metrics_goals)
        tensors = dict(**tensors_map, **tensors_goals)
        return loss_total, metrics, tensors


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
            tensors = dict(map_rec=map_pred,
                           loss_map=loss.detach(),
                           acc_map=acc_map)
            metrics = dict(loss_map=loss.mean(),
                           acc_map=nanmean(acc_map))
            if 'map_seen_mask' in obs:
                acc_map_seen = self.accuracy(map_pred, obs['map'], obs['map_seen_mask'])
                metrics['acc_map_seen'] = nanmean(acc_map_seen)

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


class GoalsProbe(nn.Module):

    def __init__(self, state_dim, conf):
        super().__init__()
        self.decoders = nn.ModuleDict({
            'goal_direction': DenseNormalDecoder(in_dim=state_dim, out_dim=2, hidden_layers=4, layer_norm=True),
            'goals_direction': DenseNormalDecoder(in_dim=state_dim, out_dim=conf.goals_size * 2, hidden_layers=4, layer_norm=True),
        })

    def training_step(self, features: TensorTBIF, obs: Dict[str, Tensor]):
        loss_total = 0
        metrics = {}
        tensors = {}
        for key, decoder in self.decoders.items():
            assert isinstance(decoder, DenseNormalDecoder)
            target = obs[key]
            _, loss, pred = decoder.training_step(features, target)
            loss_total += loss.mean()
            metrics[f'loss_{key}'] = loss.detach().mean()
            tensors[f'loss_{key}'] = loss.detach()
            tensors[f'{key}_pred'] = pred.detach()

        # Calculate MSE metrics

        with torch.no_grad():
            goals = obs['goals_direction']
            pred = tensors['goals_direction_pred']
            mse_per_coord = (goals - pred) ** 2  # (T,B,12)
            mse_per_goal = mse_per_coord.reshape(mse_per_coord.shape[:-1] + (-1, 2)).sum(-1)  # (T,B,6)
            # This is mean over all goals, average over batch
            metrics['mse_goals'] = mse_per_goal.mean(-1).mean()

            # Baseline variance, should be equal to mse_goals for a stupid model
            var_per_coord = goals.reshape((-1, goals.shape[-1])).var(0)
            var_per_goal = var_per_coord.reshape((-1, 2)).sum(-1)
            metrics['var_goals'] = var_per_goal.mean()

            log_ranges = [-1, 0, 5, 10, 50, 200, 1000]
            visage = obs.get('goals_visage')  # "visage" = "visible age" = "steps since last seen"
            if visage is not None:
                assert mse_per_goal.shape == visage.shape  # (T,B,6)
                for i in range(1, len(log_ranges)):
                    vmin = log_ranges[i-1] + 1
                    vmax = log_ranges[i]  # inclusive
                    mask = (vmin <= visage) & (visage <= vmax)
                    # This is average per goal
                    metrics[f'mse_goal_age{vmax}'] = nanmean(mse_per_goal * mask / mask)

        return loss_total, metrics, tensors


class NoProbeHead(nn.Module):

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=True)

    def training_step(self,
                      features: TensorTBIF,
                      obs: Dict[str, Tensor],
                      ):
        return torch.square(self.dummy), {}, {}
