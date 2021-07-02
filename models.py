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
                 embed_rnn=False,
                 embed_rnn_dim=512,
                 gru_layers=1
                 ):
        super().__init__()
        self._encoder: DenseEncoder = encoder
        self._decoder_image: DenseDecoder = decoder
        self._map_model: DirectHead = map_model
        self._mem_model = mem_model
        self._deter_dim = deter_dim
        self._stoch_dim = stoch_dim
        self._global_dim = mem_model.global_dim
        self._kl_weight = kl_weight
        self._map_grad = map_grad
        self._map_weight = map_weight
        self._embed_rnn = embed_rnn
        self._core = RSSMCore(embed_dim=encoder.out_dim + 2 * embed_rnn_dim if embed_rnn else encoder.out_dim,
                              action_dim=action_dim,
                              deter_dim=deter_dim,
                              stoch_dim=stoch_dim,
                              hidden_dim=hidden_dim,
                              global_dim=self._global_dim,
                              gru_layers=gru_layers)
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

    def forward(self,
                image: Tensor,     # tensor(N, B, C, H, W)
                action: Tensor,    # tensor(N, B, A)
                reset: Tensor,     # tensor(N, B)
                map_coord: Tensor,       # tensor(N, B, 4)
                in_state: Any,
                I: int = 1,
                imagine=False,     # If True, will imagine sequence, not using observations to form posterior
                do_image_pred=False,
                ):

        n, b = image.shape[:2]
        embed = self._encoder.forward(image)  # (N,B,E)

        if self._input_rnn:
            # TODO: should apply reset
            embed_rnn, _ = self._input_rnn.forward(embed, action)  # (N,B,2E)
            embed = torch.cat((embed, embed_rnn), dim=-1)  # (N,B,3E)

        # mem_out, mem_sample, mem_state = (None,), None, None
        # mem_out = self._mem_model(embed, action, reset, in_mem_state)
        # mem_sample, mem_state = mem_out[0], mem_out[-1]

        prior, post, post_samples, features, out_state = self._core.forward(embed, action, reset, in_state, None, I=I, imagine=imagine)

        image_rec = self._decoder_image.forward(features)

        
        map_features = torch.cat((features, map_coord.unsqueeze(2).expand(n, b, I, -1)), dim=-1)
        if not self._map_grad:
            map_features = map_features.detach()
        map_rec = self._map_model.forward(map_features)

        image_pred = None
        if do_image_pred:
            prior_samples = diag_normal(prior).sample()
            features_prior = self._core.feature_replace_z(features, prior_samples)
            image_pred = self._decoder_image(features_prior)

        return (
            prior,                       # (N,B,I,2S)
            post,                        # (N,B,I,2S)
            post_samples,                # (N,B,I,S)
            image_rec,                   # (N,B,I,C,H,W)
            map_rec,                     # (N,B,I,C,M,M)
            image_pred,                  # Optional[(N,B,I,C,H,W)]
            out_state,
        )

    def predict(self,
                prior, post, post_samples, image_rec, map_rec, image_pred, out_state,     # forward() output
                ):
        # Return distributions
        if image_pred is not None:
            image_pred = self._decoder_image.to_distr(image_pred)
        image_rec = self._decoder_image.to_distr(image_rec)
        map_rec = self._map_model._decoder.to_distr(map_rec)
        return (
            image_pred,    # categorical(N,B,H,W,C)
            image_rec,     # categorical(N,B,H,W,C)
            map_rec,       # categorical(N,B,HM,WM,C)
        )

    def loss(self,
             prior, post, post_samples, image_rec, map_rec, image_pred, out_state,     # forward() output
             image,                                      # tensor(N, B, C, H, W)
             map,                                        # tensor(N, B, MH, MW)
             reset,
             ):
        # Image

        image = image.unsqueeze(2).expand(image_rec.shape)
        loss_image = self._decoder_image.loss(image_rec, image)  # (N,B,I)
        logprob_img = None
        if image_pred is not None:
            logprob_img = self._decoder_image.loss(image_pred, image)

        # KL

        # loss_kl = D.kl.kl_divergence(diag_normal(post), diag_normal(prior))  # Usual VAE KL loss, only I=1

        # Sampled KL loss, works for I>1
        prior_d = diag_normal(prior)
        post_d = diag_normal(post)
        loss_kl = post_d.log_prob(post_samples) - prior_d.log_prob(post_samples)  # (N,B,I)

        # Map

        map = map.unsqueeze(2).expand(map_rec.shape)
        loss_map = self._map_model.loss(map_rec, map)    # (N,B,I)
        # metrics_map = {k.replace('loss_', 'loss_map_'): v for k, v in metrics_map.items()}  # loss_kl => loss_map_kl

        # IWAE averaging

        # loss = log (p1 + p2)/2
        #      = log (exp(l1) + exp(l2)) / 2
        #      = log (exp(l1) + exp(l2)) - log 2
        # d loss = ( exp(l1) d l1 + exp(l2) d l2 ) / (exp(l1)+exp(l2))
        # d loss / d w = exp(l1)/(exp(l1)+exp(l2)) dl1/dw + ...

        # How is loss calculated, vs logprob, why are they different?
        #
        # loss  = logavgexp_i ( log p_{i}
        #       = logavgexp_i ( sum_x ( log p_{ix}
        #       = logavgexp_i ( sum_x ( delta_{c=O(x)} ( log p_{ixc}
        #       = logavgexp_i ( sum_x ( delta_{c=O(x)} ( log softmax_c(y_{ixc})
        #       = logavgexp_i ( sum_x ( delta_{c=O(x)} ( y_{ixc} - logsumexp_c(y_{ixc})
        #
        # logprob (old incorrect way)
        #         = sum_x ( log p_{x}
        #         = sum_x ( delta_{c=O(x)} ( log p_{xc}
        #         = sum_x ( delta_{c=O(x)} ( logavgexp_i ( log p_{ixc}
        #         = sum_x ( delta_{c=O(x)} ( logavgexp_i ( y_{ixc} - logsumexp_c(y_{ixc})
        #

        # with torch.no_grad():  # This stop gradient is important for correctness
        #     weights = F.softmax(-(loss_image + loss_kl), dim=-1)    # TODO: should we apply kl_weight here?
        #     weights_map = F.softmax(-loss_map, dim=-1)
        # dloss_image = (weights * loss_image).sum(dim=-1)  # (N,B,I) => (N,B)
        # dloss_kl = (weights * loss_kl).sum(dim=-1)
        # dloss_map = (weights_map * loss_map).sum(dim=-1)

        # dloss = (dloss_image.mean()
        #          + self._kl_weight * dloss_kl.mean()
        #          + self._map_weight * dloss_map.mean())

        # Metrics

        loss_model = -logavgexp(-(self._kl_weight * loss_kl + loss_image), dim=-1)  # not the same as (loss_kl+loss_image)
        loss_map = -logavgexp(-loss_map, dim=-1)
        loss = loss_model.mean() + self._map_weight * loss_map.mean()

        with torch.no_grad():
            loss_image = -logavgexp(-loss_image, dim=-1)
            loss_kl = -logavgexp(-loss_kl, dim=-1)

            entropy_prior = prior_d.entropy().mean(dim=-1)
            entropy_post = post_d.entropy().mean(dim=-1)

            log_tensors = dict(loss_kl=loss_kl.detach(),
                               loss_image=loss_image.detach(),
                               loss_map=loss_map.detach(),
                               entropy_prior=entropy_prior,
                               entropy_post=entropy_post,
                               )
            if logprob_img is not None:
                logprob_img = -logavgexp(-logprob_img, dim=-1)  # This is *negative*-log-prob, so actually positive, same as loss
                log_tensors.update(logprob_img=logprob_img)

            metrics = dict(loss=loss.detach(),
                           loss_model=loss_model.mean(),
                           loss_model_image=loss_image.mean(),
                           loss_model_image_max=loss_image.max(),
                           loss_model_kl=loss_kl.mean(),
                           loss_model_kl_max=loss_kl.max(),
                           loss_model_mem=torch.tensor(0.0),
                           loss_map=loss_map.mean(),
                           entropy_prior=entropy_prior.mean(),
                           entropy_post=entropy_post.mean(),
                           )
            # if reset.sum() > 0:
            #     metrics.update(entropy_prior_start=(entropy_prior * reset).sum() / reset.sum())

        return loss, metrics, log_tensors


class MapPredictModel(nn.Module):

    def __init__(self, encoder, decoder, map_model, state_dim=200, action_dim=7, map_weight=1.0):
        super().__init__()
        self._encoder = encoder
        self._decoder_image = decoder
        self._map_model: DirectHead = map_model
        self._state_dim = state_dim
        self._map_weight = map_weight
        self._core = GRU2Inputs(encoder.out_dim, action_dim, mlp_dim=encoder.out_dim, state_dim=state_dim)
        self._input_rnn = None
        for m in self.modules():
            init_weights_tf2(m)

    def init_state(self, batch_size):
        return self._core.init_state(batch_size)

    def forward(self,
                image: Tensor,     # tensor(N, B, C, H, W)
                action: Tensor,    # tensor(N, B, A)
                reset: Tensor,     # tensor(N, B)
                map: Tensor,       # tensor(N, B, C, MH, MW)
                in_state: Tensor,
                I: int = 1,
                imagine=False,     # If True, will imagine sequence, not using observations to form posterior
                do_image_pred=False,
                ):

        embed = self._encoder(image)

        features, out_state = self._core.forward(embed, action, in_state)  # TODO: should apply reset

        image_rec = self._decoder_image(features)
        map_out = self._map_model.forward(features)  # NOT detached

        return (
            image_rec,                   # tensor(N, B, C, H, W)
            map_out,                     # tuple, map.forward() output
            out_state,
        )

    def predict(self,
                image_rec, map_rec, out_state,     # forward() output
                ):
        # Return distributions
        image_pred = None
        image_rec = self._decoder_image.to_distr(image_rec.unsqueeze(2))
        map_rec = self._map_model._decoder.to_distr(map_rec.unsqueeze(2))
        return (
            image_pred,    # categorical(N,B,H,W,C)
            image_rec,     # categorical(N,B,H,W,C)
            map_rec,       # categorical(N,B,HM,WM,C)
        )

    def loss(self,
             image_rec, map_out, out_state,     # forward() output
             image,                                      # tensor(N, B, C, H, W)
             map,                                        # tensor(N, B, MH, MW)
             ):
        loss_image = self._decoder_image.loss(image_rec, image)
        loss_map = self._map_model.loss(map_out, map)

        log_tensors = dict(loss_image=loss_image.detach(),
                           loss_map=loss_map.detach())

        loss_image = loss_image.mean()
        loss_map = loss_map.mean()
        loss = loss_image + self._map_weight * loss_map

        metrics = dict(loss=loss.detach(),
                       loss_model_image=loss_image.detach(),
                       loss_model=loss_image.detach(),
                       loss_map=loss_map.detach(),
                       #    **metrics_map
                       )

        return loss, metrics, log_tensors
