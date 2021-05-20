from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from modules_tools import *
from modules_rssm import *
from modules_rnn import *


class WorldModel(nn.Module):

    def __init__(self, encoder, decoder, map_model, mem_model,
                 action_dim=7,
                 deter_dim=200,
                 stoch_dim=30,
                 hidden_dim=200,
                 kl_weight=1.0,
                 map_grad=False,
                 map_weight=0.1,  # Only matters if map_grad
                 ):
        super().__init__()
        self._encoder = encoder
        self._decoder_image = decoder
        self._map_model = map_model
        self._mem_model = mem_model
        self._deter_dim = deter_dim
        self._stoch_dim = stoch_dim
        self._global_dim = mem_model.global_dim
        self._kl_weight = kl_weight
        self._map_grad = map_grad
        self._map_weight = map_weight
        self._core = RSSMCore(embed_dim=encoder.out_dim,
                              deter_dim=deter_dim,
                              stoch_dim=stoch_dim,
                              hidden_dim=hidden_dim,
                              global_dim=self._global_dim)
        self._input_rnn = GRU2Inputs(encoder.out_dim,
                                     action_dim, 
                                     encoder.out_dim, 
                                     encoder.out_dim)
        for m in self.modules():
            init_weights_tf2(m)

    def init_state(self, batch_size: int) -> Tuple[Any, Any, Any]:
        return (
            self._core.init_state(batch_size),
            self._input_rnn.init_state(batch_size),
            self._mem_model.init_state(batch_size))

    def forward(self,
                image: Tensor,     # tensor(N, B, C, H, W)
                action: Tensor,    # tensor(N, B, A)
                reset: Tensor,     # tensor(N, B)
                map: Tensor,       # tensor(N, B, C, MH, MW)
                in_state_full: Tuple[Any, Any, Any],
                ):

        in_state, in_rnn_state, in_mem_state = in_state_full
        n = image.size(0)
        embed = unflatten(self._encoder(flatten(image)), n)

        out_rnn_state = None
        # embed_rnn, out_rnn_state = self._input_rnn.forward(embed, action, in_rnn_state)  # TODO: should apply reset
        # embed = embed_rnn

        mem_out = self._mem_model(embed, action, reset, in_mem_state)
        mem_sample, mem_state = mem_out[0], mem_out[-1]

        glob_state = mem_sample
        prior, post, features, out_state = self._core(embed, action, reset, in_state, glob_state)
        features_flat = flatten(features)

        image_rec = unflatten(self._decoder_image(features_flat), n)
        map_out = self._map_model(map, features if self._map_grad else features.detach())

        return (
            prior,                       # tensor(N, B, 2*S)
            post,                        # tensor(N, B, 2*S)
            image_rec,                   # tensor(N, B, C, H, W)
            map_out,                     # tuple, map.forward() output
            features,                    # tensor(N, B, D+S+G)
            mem_out,                     # Any
            (out_state, out_rnn_state, mem_state),     # out_state_full: Any
        )

    def predict(self,
                image: Tensor,     # tensor(N, B, C, H, W)
                action: Tensor,    # tensor(N, B, A)
                reset: Tensor,     # tensor(N, B)
                map: Tensor,       # tensor(N, B, C, MH, MW)
                in_state_full: Tuple[Any, Any, Any],
                ):

        # forward() modified for prediction

        in_state, in_rnn_state, in_mem_state = in_state_full
        n = image.size(0)
        embed = unflatten(self._encoder(flatten(image)), n)

        # embed_rnn, out_rnn_state = self._input_rnn.forward(embed, action, in_rnn_state)  # TODO: should apply reset
        # embed = embed_rnn

        mem_out = self._mem_model(embed[:-1], action[:-1], reset[:-1], in_mem_state)  # Diff from forward(): hide last observation
        mem_sample, mem_state = mem_out[0], mem_out[-1]

        glob_state = mem_sample
        prior, post, features, out_state = self._core(embed, action, reset, in_state, glob_state)
        features_flat = flatten(features)

        image_rec = unflatten(self._decoder_image(features_flat), n)
        map_out = self._map_model(map, features.detach())

        # Prediction part

        # Make states with z sampled from prior instead of posterior
        h, z_post, g = features.split([self._deter_dim, self._stoch_dim, self._global_dim], -1)
        z_prior = diag_normal(prior).sample()
        features_prior = cat3(h, z_prior, g)
        image_pred = unflatten(self._decoder_image(flatten(features_prior)), n)  # (N,B,C,H,W)

        image_pred_distr = D.Categorical(logits=image_pred.permute(0, 1, 3, 4, 2))  # (N,B,C,H,W) => (N,B,H,W,C)
        image_rec_distr = D.Categorical(logits=image_rec.permute(0, 1, 3, 4, 2))
        map_rec_distr = self._map_model.predict_obs(*map_out)

        return (
            image_pred_distr,    # categorical(N,B,H,W,C)
            image_rec_distr,     # categorical(N,B,H,W,C)
            map_rec_distr,       # categorical(N,B,HM,WM,C)
        )

    def loss(self,
             prior, post, image_rec, map_out, states, mem_out, out_state_full,     # forward() output
             image,                                      # tensor(N, B, C, H, W)
             map,                                        # tensor(N, B, MH, MW)
             ):
        loss_kl = D.kl.kl_divergence(diag_normal(post), diag_normal(prior))
        loss_image = self._decoder_image.loss(image_rec, image)
        log_tensors = dict(loss_kl=loss_kl.detach())
        assert loss_kl.shape == loss_image.shape
        loss_kl = loss_kl.mean()        # (N, B) => ()
        loss_image = loss_image.mean()
        loss_mem = self._mem_model.loss(*mem_out)
        loss_model = self._kl_weight * loss_kl + loss_image + loss_mem

        loss_map, metrics_map = self._map_model.loss(*map_out, map)
        metrics_map = {k.replace('loss_', 'loss_map_'): v for k, v in metrics_map.items()}  # loss_kl => loss_map_kl

        loss = loss_model + self._map_weight * loss_map

        metrics = dict(loss_model_kl=loss_kl.detach(),
                       loss_model_image=loss_image.detach(),
                       loss_model_mem=loss_mem.detach(),
                       loss_model=loss_model.detach(),
                       loss_map=loss_map.detach(),
                       loss=loss.detach(),
                       **metrics_map)
        return loss, metrics, log_tensors


class MapPredictModel(nn.Module):

    def __init__(self, encoder, decoder, map_model, state_dim=200, action_dim=7, map_weight=1.0):
        super().__init__()
        self._encoder = encoder
        self._decoder_image = decoder
        self._map_model = map_model
        self._state_dim = state_dim
        self._map_weight = map_weight
        self._core = GRU2Inputs(encoder.out_dim, action_dim, state_dim, state_dim)
        for m in self.modules():
            init_weights_tf2(m)

    def forward(self,
                image: Tensor,     # tensor(N, B, C, H, W)
                action: Tensor,    # tensor(N, B, A)
                reset: Tensor,     # tensor(N, B)
                map: Tensor,       # tensor(N, B, C, MH, MW)
                in_state: Tensor,
                ):

        n = image.size(0)
        embed = unflatten(self._encoder(flatten(image)), n)

        features, out_state = self._core.forward(embed, action, in_state)  # TODO: should apply reset

        image_rec = unflatten(self._decoder_image(flatten(features)), n)
        map_out = self._map_model(map, features)  # NOT detached

        return (
            image_rec,                   # tensor(N, B, C, H, W)
            map_out,                     # tuple, map.forward() output
            features,                    # tensor(N, B, D+S+G)
            out_state,
        )

    def init_state(self, batch_size):
        return self._core.init_state(batch_size)

    def predict(self,
                image: Tensor,     # tensor(N, B, C, H, W)
                action: Tensor,    # tensor(N, B, A)
                reset: Tensor,     # tensor(N, B)
                map: Tensor,       # tensor(N, B, C, MH, MW)
                in_state: Tensor,
                ):

        image_rec, map_out, _, _ = self.forward(image, action, reset, map, in_state)

        image_pred_distr = D.Categorical(logits=image_rec.permute(0, 1, 3, 4, 2))  # (N,B,C,H,W) => (N,B,H,W,C)
        image_rec_distr = D.Categorical(logits=image_rec.permute(0, 1, 3, 4, 2))
        map_rec_distr = self._map_model.predict_obs(*map_out)

        return (
            image_pred_distr,    # categorical(N,B,H,W,C)
            image_rec_distr,     # categorical(N,B,H,W,C)
            map_rec_distr,       # categorical(N,B,HM,WM,C)
        )

    def loss(self,
             image_rec, map_out, features, out_state,     # forward() output
             image,                                      # tensor(N, B, C, H, W)
             map,                                        # tensor(N, B, MH, MW)
             ):
        loss_image = self._decoder_image.loss(image_rec, image)
        loss_image = loss_image.mean()
        loss_model = loss_image

        loss_map, metrics_map = self._map_model.loss(*map_out, map)
        metrics_map = {k.replace('loss_', 'loss_map_'): v for k, v in metrics_map.items()}  # loss_kl => loss_map_kl

        loss = loss_model + self._map_weight * loss_map

        log_tensors = {}
        metrics = dict(loss_model_image=loss_image.detach(),
                       loss_model=loss_model.detach(),
                       loss_map=loss_map.detach(),
                       **metrics_map)

        return loss, metrics, log_tensors
