from typing import Any, Tuple
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from modules_tools import *
from modules_rssm import *
from modules_rnn import *
from modules_io import *


class WorldModel(nn.Module):

    def __init__(self,
                 encoder,
                 decoder_image,
                 decoder_reward,
                 decoder_terminal,
                 map_model,
                 mem_model,
                 action_dim=7,
                 deter_dim=200,
                 stoch_dim=30,
                 hidden_dim=200,
                 image_weight=1.0,
                 map_grad=False,
                 map_weight=0.1,  # Only matters if map_grad
                 embed_rnn=False,
                 embed_rnn_dim=512,
                 gru_layers=1,
                 gru_type='gru'
                 ):
        super().__init__()
        self._encoder: DenseEncoder = encoder
        self._decoder_image: ConvDecoder = decoder_image
        self._decoder_reward: DenseDecoder = decoder_reward
        self._decoder_terminal: DenseDecoder = decoder_terminal
        self._map_model: DirectHead = map_model
        self._mem_model = mem_model
        self._deter_dim = deter_dim
        self._stoch_dim = stoch_dim
        self._global_dim = mem_model.global_dim
        self._image_weight = image_weight
        self._map_grad = map_grad
        self._map_weight = map_weight
        self._embed_rnn = embed_rnn
        self._core = RSSMCore(embed_dim=encoder.out_dim + 2 * embed_rnn_dim if embed_rnn else encoder.out_dim,
                              action_dim=action_dim,
                              deter_dim=deter_dim,
                              stoch_dim=stoch_dim,
                              hidden_dim=hidden_dim,
                              global_dim=self._global_dim,
                              gru_layers=gru_layers,
                              gru_type=gru_type)
        if self._embed_rnn:
            self._input_rnn = GRU2Inputs(input1_dim=encoder.out_dim,
                                         input2_dim=action_dim,
                                         mlp_dim=embed_rnn_dim,
                                         state_dim=embed_rnn_dim,
                                         bidirectional=True)
        else:
            self._input_rnn = None
        for m in self.modules():
            init_weights_tf2(m)

    def init_state(self, batch_size: int) -> Tuple[Any, Any]:
        return self._core.init_state(batch_size)

    def parameters_model(self):
        return chain(self._core.parameters(), self._encoder.parameters(), self._decoder_image.parameters())

    def parameters_map(self):
        return self._map_model.parameters()

    def forward(self,
                image: Tensor,     # tensor(N, B, C, H, W)
                reward: Tensor,
                terminal: Tensor,
                action: Tensor,    # tensor(N, B, A)
                reset: Tensor,     # tensor(N, B)
                map: Tensor,
                map_coord: Tensor,       # tensor(N, B, 4)
                in_state: Any,
                I: int = 1,
                imagine=False,     # If True, will imagine sequence, not using observations to form posterior
                do_image_pred=False,
                ):

        n, b = image.shape[:2]
        noises = torch.normal(torch.zeros((n, b, I, self._stoch_dim)),
                              torch.ones((n, b, I, self._stoch_dim))).to(image.device)  # Belongs to RSSM but need to do here for perf

        # Encoder

        # WIP: input reward, terminal

        embed = self._encoder.forward(image)  # (N,B,E)
        if self._input_rnn:
            embed_rnn, _ = self._input_rnn.forward(embed, action)  # (N,B,2E)
            embed = torch.cat((embed, embed_rnn), dim=-1)  # (N,B,3E)

        # Memory

        # mem_out, mem_sample, mem_state = (None,), None, None
        # mem_out = self._mem_model(embed, action, reset, in_mem_state)
        # mem_sample, mem_state = mem_out[0], mem_out[-1]

        # RSSM

        prior, post, post_samples, features, out_state = self._core.forward(embed, action, reset, in_state, None, noises, I=I, imagine=imagine)

        # Decoder

        image_rec = self._decoder_image.forward(features)
        reward_rec = self._decoder_reward.forward(features)
        terminal_rec = self._decoder_terminal.forward(features)

        # Predictions

        image_pred, reward_pred, terminal_pred = None, None, None
        if do_image_pred:
            prior_samples = diag_normal(prior).sample()
            features_prior = self._core.feature_replace_z(features, prior_samples)
            image_pred = self._decoder_image(features_prior)
            reward_pred = self._decoder_reward(features_prior)
            terminal_pred = self._decoder_terminal(features_prior)

        # Map

        map_coord = map_coord.unsqueeze(2).expand(n, b, I, -1)
        map = map.unsqueeze(2).expand(n, b, I, *map.shape[2:])
        map_features = torch.cat((features, map_coord), dim=-1)
        if not self._map_grad:
            map_features = map_features.detach()
        map_out = self._map_model.forward(map_features, map, do_image_pred=do_image_pred)

        return (
            prior,                       # (N,B,I,2S)
            post,                        # (N,B,I,2S)
            post_samples,                # (N,B,I,S)
            image_rec,                   # (N,B,I,C,H,W)
            reward_rec,
            terminal_rec,
            map_out,                     # (N,B,I,C,M,M)
            image_pred,                  # Optional[(N,B,I,C,H,W)]
            reward_pred,
            terminal_pred,
            out_state,
        )

    def predict(self,
                prior, post, post_samples, image_rec, reward_rec, terminal_rec, map_out, image_pred, reward_pred, terminal_pred, out_state,     # forward() output
                ):
        # Return distributions
        if image_pred is not None:
            image_pred = self._decoder_image.to_distr(image_pred)
        image_rec = self._decoder_image.to_distr(image_rec)
        map_rec = self._map_model.to_distr(*map_out)
        return (
            image_pred,    # categorical(N,B,H,W,C)
            image_rec,     # categorical(N,B,H,W,C)
            map_rec,       # categorical(N,B,HM,WM,C)
        )

    def loss(self,
             prior, post, post_samples, image_rec, reward_rec, terminal_rec, map_out, image_pred, reward_pred, terminal_pred, out_state,     # forward() output
             image, reward, terminal,
             map,                                        # tensor(N, B, MH, MW)
             reset,
             map_coord
             ):
        N, B, I = image_rec.shape[:3]
        image = image.unsqueeze(2).expand(N, B, I, *image.shape[2:])
        reward = reward.unsqueeze(2).expand(N, B, I, *reward.shape[2:])
        terminal = terminal.unsqueeze(2).expand(N, B, I, *terminal.shape[2:])

        # Reconstruction

        loss_image = self._decoder_image.loss(image_rec, image)  # (N,B,I)
        loss_reward = self._decoder_reward.loss(reward_rec, reward)  # (N,B,I)
        loss_terminal = self._decoder_terminal.loss(terminal_rec, terminal)  # (N,B,I)

        # WIP: loss_reward, loss_terminal

        # KL

        prior_d = diag_normal(prior)
        post_d = diag_normal(post)
        if I == 1:
            # Analytic KL loss, standard for VAE
            loss_kl = loss_kl_metric = D.kl.kl_divergence(post_d, prior_d)  # (N,B,I)
        else:
            # Sampled KL loss, for IWAE
            loss_kl = post_d.log_prob(post_samples) - prior_d.log_prob(post_samples)
            # Log analytic KL loss for metrics, it's nicer and avoids negative values
            loss_kl_metric = D.kl.kl_divergence(post_d, prior_d)

        # Map

        map = map.unsqueeze(2).expand(N, B, I, *map.shape[2:])  # TODO: include in map_out. or even merge forward+loss => train_step
        loss_map, metrics_map = self._map_model.loss(*map_out, map, map_coord)    # type: ignore

        # Total loss

        assert loss_kl.shape == loss_image.shape == loss_reward.shape == loss_terminal.shape
        loss_model = -logavgexp(-(
            loss_kl
            + self._image_weight * loss_image
            + loss_reward
            + loss_terminal
        ), dim=-1)
        loss_map = -logavgexp(-loss_map, dim=-1)
        loss = loss_model.mean() + self._map_weight * loss_map.mean()

        # IWAE according to paper

        # with torch.no_grad():
        #     weights = F.softmax(-(loss_image + loss_kl), dim=-1)
        #     weights_map = F.softmax(-loss_map, dim=-1)
        # dloss_image = (weights * loss_image).sum(dim=-1)
        # dloss_kl = (weights * loss_kl).sum(dim=-1)
        # dloss_map = (weights_map * loss_map).sum(dim=-1)
        # dloss = (self._image_weight * dloss_image.mean()
        #          + dloss_kl.mean()
        #          + self._map_weight * dloss_map.mean())

        # Metrics

        with torch.no_grad():
            loss_image = -logavgexp(-loss_image, dim=-1)
            loss_reward = -logavgexp(-loss_reward, dim=-1)
            loss_terminal = -logavgexp(-loss_terminal, dim=-1)
            loss_kl = -logavgexp(-loss_kl_metric, dim=-1)
            entropy_prior = prior_d.entropy().mean(dim=-1)
            entropy_post = post_d.entropy().mean(dim=-1)

            log_tensors = dict(loss_kl=loss_kl.detach(),
                               loss_image=loss_image.detach(),
                               loss_map=loss_map.detach(),
                               entropy_prior=entropy_prior,
                               entropy_post=entropy_post,
                               )

            metrics = dict(loss=loss.detach(),
                           loss_model=loss_model.mean(),
                           loss_model_image=loss_image.mean(),
                           loss_model_image_max=loss_image.max(),
                           loss_model_reward=loss_reward.mean(),
                           loss_model_terminal=loss_terminal.mean(),
                           loss_model_kl=loss_kl.mean(),
                           loss_model_kl_max=loss_kl.max(),
                           loss_map=loss_map.mean(),
                           entropy_prior=entropy_prior.mean(),
                           entropy_post=entropy_post.mean(),
                           )

            if image_pred is not None:
                logprob_img = self._decoder_image.loss(image_pred, image)
                logprob_img = -logavgexp(-logprob_img, dim=-1)  # This is *negative*-log-prob, so actually positive, same as loss
                log_tensors.update(logprob_img=logprob_img)

            metrics.update(**metrics_map)

        return loss, metrics, log_tensors


# class MapPredictModel(nn.Module):

#     def __init__(self, encoder, decoder, map_model, state_dim=200, action_dim=7, map_weight=1.0):
#         super().__init__()
#         self._encoder = encoder
#         self._decoder_image = decoder
#         self._map_model: DirectHead = map_model
#         self._state_dim = state_dim
#         self._map_weight = map_weight
#         self._core = GRU2Inputs(encoder.out_dim, action_dim, mlp_dim=encoder.out_dim, state_dim=state_dim)
#         self._input_rnn = None
#         for m in self.modules():
#             init_weights_tf2(m)

#     def init_state(self, batch_size):
#         return self._core.init_state(batch_size)

#     def forward(self,
#                 image: Tensor,     # tensor(N, B, C, H, W)
#                 action: Tensor,    # tensor(N, B, A)
#                 reset: Tensor,     # tensor(N, B)
#                 map: Tensor,       # tensor(N, B, C, MH, MW)
#                 in_state: Tensor,
#                 I: int = 1,
#                 imagine=False,     # If True, will imagine sequence, not using observations to form posterior
#                 do_image_pred=False,
#                 ):

#         embed = self._encoder(image)

#         features, out_state = self._core.forward(embed, action, in_state)  # TODO: should apply reset

#         image_rec = self._decoder_image(features)
#         map_out = self._map_model.forward(features)  # NOT detached

#         return (
#             image_rec,                   # tensor(N, B, C, H, W)
#             map_out,                     # tuple, map.forward() output
#             out_state,
#         )

#     def predict(self,
#                 image_rec, map_rec, out_state,     # forward() output
#                 ):
#         # Return distributions
#         image_pred = None
#         image_rec = self._decoder_image.to_distr(image_rec.unsqueeze(2))
#         map_rec = self._map_model._decoder.to_distr(map_rec.unsqueeze(2))
#         return (
#             image_pred,    # categorical(N,B,H,W,C)
#             image_rec,     # categorical(N,B,H,W,C)
#             map_rec,       # categorical(N,B,HM,WM,C)
#         )

#     def loss(self,
#              image_rec, map_out, out_state,     # forward() output
#              image,                                      # tensor(N, B, C, H, W)
#              map,                                        # tensor(N, B, MH, MW)
#              ):
#         loss_image = self._decoder_image.loss(image_rec, image)
#         loss_map = self._map_model.loss(map_out, map)

#         log_tensors = dict(loss_image=loss_image.detach(),
#                            loss_map=loss_map.detach())

#         loss_image = loss_image.mean()
#         loss_map = loss_map.mean()
#         loss = loss_image + self._map_weight * loss_map

#         metrics = dict(loss=loss.detach(),
#                        loss_model_image=loss_image.detach(),
#                        loss_model=loss_image.detach(),
#                        loss_map=loss_map.detach(),
#                        #    **metrics_map
#                        )

#         return loss, metrics, log_tensors
