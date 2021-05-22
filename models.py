from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from modules_tools import *
from modules_rssm import *
from modules_rnn import *
from modules_io import *


class WorldModel(nn.Module):

    def __init__(self, encoder, decoder, map_model, mem_model,
                 action_dim=7,
                 deter_dim=200,
                 stoch_dim=30,
                 hidden_dim=200,
                 kl_weight=1.0,
                 map_grad=False,
                 map_weight=0.1,  # Only matters if map_grad
                 iwae_samples=0,      # arxiv.org/abs/1509.00519
                 ):
        super().__init__()
        self._encoder = encoder
        self._decoder_image: DenseDecoder = decoder
        self._map_model: DirectHead = map_model
        self._mem_model = mem_model
        self._deter_dim = deter_dim
        self._stoch_dim = stoch_dim
        self._global_dim = mem_model.global_dim
        self._kl_weight = kl_weight
        self._map_grad = map_grad
        self._map_weight = map_weight
        self._iwae_samples = iwae_samples
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
                I: int = 1
                ):

        in_state, in_rnn_state, in_mem_state = in_state_full
        n, b = image.shape[:2]
        embed = unflatten(self._encoder(flatten(image)), n)  # (N,B,E)

        out_rnn_state = None
        # embed_rnn, out_rnn_state = self._input_rnn.forward(embed, action, in_rnn_state)  # TODO: should apply reset
        # embed = embed_rnn

        mem_out, mem_sample, mem_state = (None,), None, None
        # mem_out = self._mem_model(embed, action, reset, in_mem_state)
        # mem_sample, mem_state = mem_out[0], mem_out[-1]

        prior, post, post_samples, features, out_state = self._core.forward(embed, action, reset, in_state, mem_sample, I=I)
        features_flat = flatten3(features)

        image_rec = unflatten3(self._decoder_image.forward(features_flat), (n, b))
        map_out = unflatten3(self._map_model.forward(features_flat if self._map_grad else features_flat.detach()), (n, b))

        return (
            prior,                       # (N,B,I,2S)
            post,                        # (N,B,I,2S)
            post_samples,                # (N,B,I,S)
            image_rec,                   # (N,B,I,C,H,W)
            map_out,                     # (N,B,I,C,M,M)
            features,                    # (N,B,I,D+S+G)
            mem_out,                     # Any
            (out_state, out_rnn_state, mem_state),     # out_state_full: Any
        )

    def predict(self,
                image: Tensor,     # tensor(N, B, C, H, W)
                action: Tensor,    # tensor(N, B, A)
                reset: Tensor,     # tensor(N, B)
                map: Tensor,       # tensor(N, B, C, MH, MW)
                in_state_full: Tuple[Any, Any, Any],
                I: int = 1
                ):

        # forward() modified for prediction

        in_state, in_rnn_state, in_mem_state = in_state_full
        n, b = image.shape[:2]
        embed = unflatten(self._encoder(flatten(image)), n)  # (N,B,E)

        out_rnn_state = None
        # embed_rnn, out_rnn_state = self._input_rnn.forward(embed, action, in_rnn_state)  # TODO: should apply reset
        # embed = embed_rnn

        mem_out, mem_sample, mem_state = (None,), None, None
        # mem_out = self._mem_model(embed[:-1], action[:-1], reset[:-1], in_mem_state)  # Diff from forward(): hide last observation
        # mem_sample, mem_state = mem_out[0], mem_out[-1]

        prior, post, post_samples, features, out_state = self._core.forward(embed, action, reset, in_state, mem_sample, I=I)
        features_flat = flatten3(features)

        image_rec = unflatten3(self._decoder_image.forward(features_flat), (n, b))
        map_out = unflatten3(self._map_model.forward(features_flat if self._map_grad else features_flat.detach()), (n, b))

        # Prediction part

        # Make states with z sampled from prior instead of posterior
        h, post_samples, g = features.split([self._deter_dim, self._stoch_dim, self._global_dim], -1)
        prior_samples = diag_normal(prior).sample()
        features_prior = cat3(h, prior_samples, g)
        image_pred = unflatten3(self._decoder_image(flatten3(features_prior)), (n, b))

        image_pred_distr = imgrec_to_distr(image_pred)
        image_rec_distr = imgrec_to_distr(image_rec)
        map_rec_distr = self._map_model.predict_obs(map_out)

        return (
            image_pred_distr,    # categorical(N,B,H,W,C)
            image_rec_distr,     # categorical(N,B,H,W,C)
            map_rec_distr,       # categorical(N,B,HM,WM,C)
        )

    def loss(self,
             prior, post, post_samples, image_rec, map_out, states, mem_out, out_state_full,     # forward() output
             image,                                      # tensor(N, B, C, H, W)
             map,                                        # tensor(N, B, MH, MW)
             ):
        metrics, log_tensors = {}, {}

        # Image

        N, B, I = image_rec.shape[:3]
        output = flatten3(image_rec)  # (N,B,I,...) => (NBI,...)
        target = flatten3(image.unsqueeze(2).expand(image_rec.shape))
        loss_image = self._decoder_image.loss(output, target)
        loss_image = unflatten3(loss_image, (N, B))  # (N,B,I)

        # KL

        # # Usual VAE KL loss
        # loss_kl = D.kl.kl_divergence(diag_normal(post), diag_normal(prior))
        # Sampled KL loss
        prior_d = diag_normal(prior)
        post_d = diag_normal(post)
        loss_kl = post_d.log_prob(post_samples) - prior_d.log_prob(post_samples)  # (N,B,I)

        # Map

        map_rec = map_out  # TODO
        output = flatten3(map_rec)  # (N,B,I,...) => (NBI,...)
        target = flatten3(map.unsqueeze(2).expand(map_rec.shape))
        loss_map = self._map_model.loss(output, target)  # TODO
        # metrics_map = {k.replace('loss_', 'loss_map_'): v for k, v in metrics_map.items()}  # loss_kl => loss_map_kl
        loss_map = unflatten3(loss_map, (N, B))

        # IWAE averaging

        with torch.no_grad():
            weights = F.softmax(- (loss_image + loss_kl), dim=-1)
            weights_map = F.softmax(-loss_map, dim=-1)
        loss_image = (weights * loss_image).sum(dim=-1)  # (N,B,I) => (N,B)
        loss_kl = (weights * loss_kl).sum(dim=-1)
        loss_map = (weights_map * loss_map).sum(dim=-1)

        # Add up

        log_tensors.update(loss_kl=loss_kl.detach(), loss_image=loss_image.detach())
        metrics.update(loss_model_kl_max=loss_kl.detach().max(), loss_model_image_max=loss_image.detach().max())

        assert loss_kl.shape == loss_image.shape
        loss_kl = loss_kl.mean()        # (N, B) => ()
        loss_image = loss_image.mean()
        loss_mem = self._mem_model.loss(*mem_out)
        loss_model = self._kl_weight * loss_kl + loss_image + loss_mem

        loss_map = loss_map.mean()
        loss = loss_model + self._map_weight * loss_map

        metrics.update(loss_model_kl=loss_kl.detach(),
                       loss_model_image=loss_image.detach(),
                       loss_model_mem=loss_mem.detach(),
                       loss_model=loss_model.detach(),
                       loss_map=loss_map.detach(),
                       loss=loss.detach(),
                    #    **metrics_map
                       )
        return loss, metrics, log_tensors


class MapPredictModel(nn.Module):

    def __init__(self, encoder, decoder, map_model, state_dim=200, action_dim=7, map_weight=1.0):
        super().__init__()
        self._encoder = encoder
        self._decoder_image = decoder
        self._map_model: DirectHead = map_model
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
        map_out = self._map_model.forward(features)  # NOT detached

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

        loss_map = self._map_model.loss(map_out, map)
        # metrics_map = {k.replace('loss_', 'loss_map_'): v for k, v in metrics_map.items()}  # loss_kl => loss_map_kl

        loss = loss_model + self._map_weight * loss_map

        log_tensors = {}
        metrics = dict(loss_model_image=loss_image.detach(),
                       loss_model=loss_model.detach(),
                       loss_map=loss_map.detach(),
                    #    **metrics_map
                       )

        return loss, metrics, log_tensors
