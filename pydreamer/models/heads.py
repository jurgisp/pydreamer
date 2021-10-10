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


class DirectHead(nn.Module):

    def __init__(self, decoder):
        super().__init__()
        self._decoder: DenseDecoder = decoder

    def forward(self, state, obs, do_image_pred=False):
        obs_pred = self._decoder.forward(state)
        return (obs_pred, )

    def loss(self, obs_pred: TensorNBICHW, obs_target: TensorNBICHW, map_coord: TensorNBI4, map_seen_mask: Tensor):
        loss = self._decoder.loss(obs_pred, obs_target)  # (N,B,I)
        loss = -logavgexp(-loss, dim=-1)  # (N,B,I) => (N,B)
        with torch.no_grad():
            acc_map = self._decoder.accuracy(obs_pred, obs_target, map_coord)
            acc_map_seen = self._decoder.accuracy(obs_pred, obs_target, map_coord, map_seen_mask)
            tensors = dict(loss_map=loss.detach(),
                           acc_map=acc_map)
            metrics = dict(loss_map=loss.mean(),
                           acc_map=nanmean(acc_map),
                           acc_map_seen=nanmean(acc_map_seen))
        return loss.mean(), metrics, tensors

    def predict(self, obs_pred) -> Tensor:
        # TODO: obsolete this method
        return self._decoder.to_distr(obs_pred)


class VAEHead(nn.Module):
    # Conditioned VAE

    def __init__(self, encoder: ConvEncoder, decoder: ConvDecoder, state_dim=230, hidden_dim=200, latent_dim=30):
        super().__init__()
        self._encoder = encoder
        self._decoder = decoder
        assert decoder.in_dim == state_dim + latent_dim

        self._prior_mlp = nn.Sequential(nn.Linear(state_dim, hidden_dim),
                                        nn.ELU(),
                                        nn.Linear(hidden_dim, 2 * latent_dim))

        self._post_mlp = nn.Sequential(nn.Linear(state_dim + encoder.out_dim, hidden_dim),
                                       nn.ELU(),
                                       nn.Linear(hidden_dim, 2 * latent_dim))

    def forward(self, state, obs, do_image_pred=False):
        embed = self._encoder.forward(obs)
        prior = self._prior_mlp(state)
        post = self._post_mlp(torch.cat([state, embed], -1))
        sample = diag_normal(post).rsample()
        obs_rec = self._decoder(torch.cat([state, sample], -1))
        if do_image_pred:
            prior_sample = diag_normal(prior).sample()
            obs_pred = self._decoder(torch.cat([state, prior_sample], -1))
        else:
            obs_pred = None
        return obs_rec, prior, post, obs_pred

    def loss(self,
             obs_rec, prior, post, obs_pred,  # forward() output
             obs_target, map_coord, map_seen_mask
             ):
        prior_d = diag_normal(prior)
        post_d = diag_normal(post)
        loss_kl = D.kl.kl_divergence(post_d, prior_d)
        loss_rec = self._decoder.loss(obs_rec, obs_target)
        assert loss_kl.shape == loss_rec.shape
        loss = loss_kl + loss_rec  # (N,B,I)
        loss = -logavgexp(-loss, dim=-1)  # (N,B,I) => (N,B)

        with torch.no_grad():
            loss_rec = -logavgexp(-loss_rec, dim=-1)
            loss_kl = -logavgexp(-loss_kl, dim=-1)
            entropy_prior = prior_d.entropy().mean(dim=-1)
            entropy_post = post_d.entropy().mean(dim=-1)
            tensors = dict(loss_map=loss.detach())
            metrics = dict(loss_map=loss.mean(),
                           loss_map_rec=loss_rec.mean(),
                           loss_map_kl=loss_kl.mean(),
                           entropy_map_prior=entropy_prior.mean(),
                           entropy_map_post=entropy_post.mean(),
                           )
            if obs_pred is not None:
                acc_map = self._decoder.accuracy(obs_pred, obs_target, map_coord)
                tensors.update(acc_map=acc_map)
                metrics.update(acc_map=nanmean(acc_map))

        return loss.mean(), metrics, tensors

    def predict(self, obs_rec, prior, post, obs_pred) -> Tensor:
        # TODO: obsolete this method
        if obs_pred is not None:
            return self._decoder.to_distr(obs_pred)
        else:
            return self._decoder.to_distr(obs_rec)


class NoHead(nn.Module):

    def __init__(self, out_shape):
        super().__init__()
        self.out_shape = out_shape  # (C,MH,MW)
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, state, obs, do_image_pred=False):
        return (obs,)

    def loss(self, obs_pred: TensorNBICHW, obs_target: TensorNBICHW, map_coord: TensorNBI4, map_seen_mask: Tensor):
        return torch.square(self._dummy), {}, {}

    def predict(self, output):
        # TODO: obsolete this method
        assert len(output.shape) == 6  # (N,B,I,C,H,W)
        return output.mean(dim=2)  # (N,B,I,C,H,W) => (N,B,C,H,W)
