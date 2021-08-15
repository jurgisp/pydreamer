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
from modules_mem import *
from modules_rl import *


class Dreamer(nn.Module):

    def __init__(self, conf):
        super().__init__()

        state_dim = conf.deter_dim + conf.stoch_dim + conf.global_dim
        n_map_coords = 4
        map_state_dim = state_dim + n_map_coords

        # Map decoder

        if conf.map_model == 'vae':
            if conf.map_decoder == 'cnn':
                map_model = VAEHead(
                    encoder=ConvEncoder(in_channels=conf.map_channels,
                                        out_dim=conf.embed_dim),
                    decoder=ConvDecoder(in_dim=map_state_dim + conf.map_stoch_dim,
                                        mlp_layers=2,
                                        out_channels=conf.map_channels),
                    state_dim=map_state_dim,
                    latent_dim=conf.map_stoch_dim,
                    hidden_dim=conf.hidden_dim
                )
            else:
                raise NotImplementedError

        elif conf.map_model == 'direct':
            if conf.map_decoder == 'cnn':
                map_model = DirectHead(
                    decoder=ConvDecoder(in_dim=map_state_dim,
                                        mlp_layers=2,
                                        out_channels=conf.map_channels))

            else:
                map_model = DirectHead(
                    decoder=DenseDecoder(in_dim=map_state_dim,
                                         out_shape=(conf.map_channels, conf.map_size, conf.map_size),
                                         hidden_dim=conf.map_hidden_dim,
                                         hidden_layers=conf.map_hidden_layers))    # type: ignore
        elif conf.map_model == 'none':
            map_model = NoHead(out_shape=(conf.map_channels, conf.map_size, conf.map_size))
        else:
            assert False, f'Unknown map_model={conf.map_model}'

        self.map_model: DirectHead = map_model  # type: ignore

        # World model

        self.wm = WorldModel(
            conf,
            action_dim=conf.action_dim,
            deter_dim=conf.deter_dim,
            stoch_dim=conf.stoch_dim,
            hidden_dim=conf.hidden_dim,
            image_weight=conf.image_weight,
            embed_rnn=conf.embed_rnn != 'none',
            gru_layers=conf.gru_layers,
            gru_type=conf.gru_type
        )

        # Actor critic

        self.ac = ActorCritic(in_dim=state_dim,
                              out_actions=conf.action_dim,
                              discount=conf.discount,
                              discount_lambda=conf.discount_lambda,
                              temperature=conf.actor_ent,
                              target_interval=conf.target_interval,
                              )

    def forward(self,
                image: TensorNBCHW,   # (1,B,C,H,W)
                prev_reward: Tensor,  # (1,B)
                prev_action: Tensor,  # (1,B,A)
                reset: Tensor,        # (1,B)
                in_state: Any,
                ):
        N, B = image.shape[:2]
        assert N == 1

        # Forward (world model)

        terminal = torch.zeros_like(reset)
        output = self.wm.forward(image, prev_reward, terminal, prev_action, reset, in_state)
        features = output[0]
        out_state = output[-1]

        # Forward (actor critic)

        feature = features[0, :, 0]  # (N=1,B,I=1,F) => (B,F)
        action_p = self.ac.forward_act(feature)

        return action_p, out_state

    def train(self,
              image: TensorNBCHW,
              reward: Tensor,
              terminal: Tensor,
              action: Tensor,     # (N,B,A)
              reset: Tensor,      # (N,B)
              map: Tensor,
              map_coord: Tensor,  # (N,B,4)
              in_state: Any,
              I: int = 1,         # IWAE samples
              H: int = 1,        # Imagination horizon
              imagine=False,      # If True, will imagine sequence, not using observations to form posterior
              do_image_pred=False,
              do_output_tensors=False
              ):
        N, B = image.shape[:2]

        # Forward (world model)

        output = self.wm.forward(image, reward, terminal, action, reset, in_state, I, imagine, do_image_pred)
        features = output[0]
        states = output[-2]
        out_state = output[-1]

        # Forward (map)

        map_coord = map_coord.unsqueeze(2).expand(N, B, I, -1)  # TODO: move inside map_model.forward()
        map = map.unsqueeze(2).expand(N, B, I, *map.shape[2:])
        map_features = torch.cat((features, map_coord), dim=-1)
        map_features = map_features.detach()
        map_out = self.map_model.forward(map_features, map, do_image_pred=do_image_pred)

        # Forward (actor critic)

        with torch.no_grad():  # Not using dynamics gradients for now, just Reinforce
            in_state_dream: StateB = map_structure(states, lambda x: flatten_batch(x.detach())[0])  # type: ignore
            features_dream, actions_dream = self.dream(in_state_dream, H)                       # (H+1,NBI,D)
            rewards_dream = self.wm._decoder_reward.forward(features_dream)      # (H+1,NBI,2)
            terminals_dream = self.wm._decoder_terminal.forward(features_dream)  # (H+1,NBI,2)

        # Loss

        loss_model, metrics, loss_tensors = self.wm.loss(*output, image, reward, terminal, reset)
        metrics.update(loss=metrics['loss_wm'])

        loss_map, metrics_map, loss_tensors_map = self.map_model.loss(*map_out, map, map_coord)    # type: ignore
        metrics.update(**metrics_map)
        loss_tensors.update(**loss_tensors_map)

        loss_ac, metrics_ac, loss_tensors_ac = self.ac.train(features_dream, rewards_dream, terminals_dream, actions_dream)
        metrics.update(**metrics_ac)
        loss_tensors_ac = {k: unflatten_batch(v, (N, B, I)).mean(dim=-1) for k, v in loss_tensors_ac.items()}  # (N,B)
        loss_tensors.update(**loss_tensors_ac)

        # Predict

        if do_output_tensors:
            with torch.no_grad():
                image_pred, image_rec = self.wm.predict(*output)
                map_rec = self.map_model.to_distr(*map_out)
                out_tensors = (image_pred, image_rec, map_rec)
        else:
            out_tensors = None

        losses = (loss_model, loss_map, loss_ac)
        return losses, metrics, loss_tensors, out_state, out_tensors

    def dream(self, in_state: StateB, H: int):
        NBI = len(in_state[0])  # Imagine batch size = N*B*I
        noises = self.wm._core.generate_noises(H, (NBI, ), in_state[0].device)
        features = []
        actions = []
        state = in_state

        for i in range(H):
            feature = self.wm._core.to_feature(*state)
            action = self.ac.forward_act(feature).sample()
            features.append(feature)
            actions.append(action)
            _, state = self.wm._core._cell.forward_prior(action, state, noises[i])

        feature = self.wm._core.to_feature(*state)
        features.append(feature)

        features = torch.stack(features)  # (H+1,NBI,D)
        actions = torch.stack(actions)  # (H,NBI,A)
        return features, actions


class WorldModel(nn.Module):

    def __init__(self,
                 conf,
                 action_dim=7,
                 deter_dim=200,
                 stoch_dim=30,
                 hidden_dim=200,
                 image_weight=1.0,
                 embed_rnn=False,
                 embed_rnn_dim=512,
                 gru_layers=1,
                 gru_type='gru'
                 ):
        super().__init__()
        self._deter_dim = deter_dim
        self._stoch_dim = stoch_dim
        self._global_dim = 0
        self._image_weight = image_weight
        self._embed_rnn = embed_rnn
        self._mem_model = NoMemory()

        # Encoder

        encoder_channels = conf.image_channels + 2  # + reward, terminal
        if conf.image_encoder == 'cnn':
            self._encoder = ConvEncoder(in_channels=encoder_channels,
                                        out_dim=conf.embed_dim)
        else:
            self._encoder = DenseEncoder(in_dim=conf.image_size * conf.image_size * encoder_channels,
                                         out_dim=conf.embed_dim,
                                         hidden_layers=conf.image_encoder_layers)

        if self._embed_rnn:
            self._input_rnn = GRU2Inputs(input1_dim=self._encoder.out_dim,
                                         input2_dim=action_dim,
                                         mlp_dim=embed_rnn_dim,
                                         state_dim=embed_rnn_dim,
                                         bidirectional=True)
        else:
            self._input_rnn = None

        # Decoders

        state_dim = conf.deter_dim + conf.stoch_dim + conf.global_dim
        if conf.image_decoder == 'cnn':
            self._decoder_image = ConvDecoder(in_dim=state_dim,
                                              out_channels=conf.image_channels)
        else:
            self._decoder_image = DenseDecoder(in_dim=state_dim,
                                               out_shape=(conf.image_channels, conf.image_size, conf.image_size),
                                               hidden_layers=conf.image_decoder_layers,
                                               min_prob=conf.image_decoder_min_prob)

        self._decoder_reward = DenseNormalHead(in_dim=state_dim, hidden_layers=conf.reward_decoder_layers)
        self._decoder_terminal = DenseBernoulliHead(in_dim=state_dim, hidden_layers=conf.terminal_decoder_layers)

        # Memory model

        # if conf.mem_model == 'global_state':
        #     mem_model = GlobalStateMem(embed_dim=conf.embed_dim,
        #                             action_dim=conf.action_dim,
        #                             mem_dim=conf.deter_dim,
        #                             stoch_dim=conf.global_dim,
        #                             hidden_dim=conf.hidden_dim,
        #                             loss_type=conf.mem_loss_type)
        # else:
        #     mem_model = NoMemory()

        # RSSM

        self._core = RSSMCore(embed_dim=self._encoder.out_dim + 2 * embed_rnn_dim if embed_rnn else self._encoder.out_dim,
                              action_dim=action_dim,
                              deter_dim=deter_dim,
                              stoch_dim=stoch_dim,
                              hidden_dim=hidden_dim,
                              global_dim=self._global_dim,
                              gru_layers=gru_layers,
                              gru_type=gru_type)

        # Init

        for m in self.modules():
            init_weights_tf2(m)

    def init_state(self, batch_size: int) -> Tuple[Any, Any]:
        return self._core.init_state(batch_size)

    def forward(self,
                image: TensorNBCHW,
                reward: TensorNB,
                terminal: TensorNB,
                action: Tensor,    # tensor(N, B, A)
                reset: Tensor,     # tensor(N, B)
                in_state: Any,
                I: int = 1,
                imagine=False,     # If True, will imagine sequence, not using observations to form posterior
                do_image_pred=False,
                ):

        N, B, C, H, W = image.shape
        noises = self._core.generate_noises(N, (B * I, ), image.device)  # Belongs to RSSM but need to do here for perf

        # Encoder

        reward_plane = reward.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand((N, B, 1, H, W))
        terminal_plane = terminal.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand((N, B, 1, H, W))
        observation = torch.cat([  # (N,B,C+2,H,W)
            image,
            reward_plane.to(image.dtype),
            terminal_plane.to(image.dtype)], dim=-3)
        embed = self._encoder.forward(observation)  # (N,B,E)
        if self._input_rnn:
            embed_rnn, _ = self._input_rnn.forward(embed, action)  # (N,B,2E)
            embed = torch.cat((embed, embed_rnn), dim=-1)  # (N,B,3E)

        # Memory

        # mem_out, mem_sample, mem_state = (None,), None, None
        # mem_out = self._mem_model(embed, action, reset, in_mem_state)
        # mem_sample, mem_state = mem_out[0], mem_out[-1]

        # RSSM

        prior, post, post_samples, features, states, out_state = self._core.forward(embed, action, reset, in_state, None, noises, I=I, imagine=imagine)

        # Decoder

        image_rec = self._decoder_image.forward(features)
        reward_rec = self._decoder_reward.forward(features)
        terminal_rec = self._decoder_terminal.forward(features)

        # Predictions

        image_pred, reward_pred, terminal_pred = None, None, None
        if do_image_pred:
            prior_samples = diag_normal(prior).sample()
            features_prior = self._core.feature_replace_z(features, prior_samples)
            image_pred = self._decoder_image.forward(features_prior)
            reward_pred = self._decoder_reward.forward(features_prior)
            terminal_pred = self._decoder_terminal.forward(features_prior)

        return (
            features,
            prior,                       # (N,B,I,2S)
            post,                        # (N,B,I,2S)
            post_samples,                # (N,B,I,S)
            image_rec,                   # (N,B,I,C,H,W)
            reward_rec,
            terminal_rec,
            image_pred,                  # Optional[(N,B,I,C,H,W)]
            reward_pred,
            terminal_pred,
            states,
            out_state,
        )

    def predict(self,
                features, prior, post, post_samples, image_rec, reward_rec, terminal_rec, image_pred, reward_pred, terminal_pred, states, out_state,     # forward() output
                ):
        if image_pred is not None:
            image_pred = self._decoder_image.to_distr(image_pred)
        image_rec = self._decoder_image.to_distr(image_rec)
        return (
            image_pred,    # categorical(N,B,H,W,C)
            image_rec,     # categorical(N,B,H,W,C)
        )

    def loss(self,
             features, prior, post, post_samples, image_rec, reward_rec, terminal_rec, image_pred, reward_pred, terminal_pred, states, out_state,     # forward() output
             image, reward, terminal,
             reset,
             ):
        N, B, I = image_rec.shape[:3]
        image = image.unsqueeze(2).expand(N, B, I, *image.shape[2:])
        reward = reward.unsqueeze(2).expand(N, B, I, *reward.shape[2:])
        terminal = terminal.unsqueeze(2).expand(N, B, I, *terminal.shape[2:])

        # Reconstruction

        loss_image = self._decoder_image.loss(image_rec, image)  # (N,B,I)
        loss_reward = self._decoder_reward.loss(reward_rec, reward)  # (N,B,I)
        loss_terminal = self._decoder_terminal.loss(terminal_rec, terminal)  # (N,B,I)

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

        # Total loss

        assert loss_kl.shape == loss_image.shape == loss_reward.shape == loss_terminal.shape
        loss_model = -logavgexp(-(
            loss_kl
            + self._image_weight * loss_image
            + loss_reward
            + loss_terminal
        ), dim=-1)

        # IWAE according to paper

        # with torch.no_grad():
        #     weights = F.softmax(-(loss_image + loss_kl), dim=-1)
        # dloss = (weights * (loss_image + loss_kl)).sum(dim=-1).mean()

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
                               entropy_prior=entropy_prior,
                               entropy_post=entropy_post,
                               )

            metrics = dict(loss_wm=loss_model.mean(),
                           loss_wm_image=loss_image.mean(),
                           loss_wm_image_max=loss_image.max(),
                           loss_wm_reward=loss_reward.mean(),
                           loss_wm_terminal=loss_terminal.mean(),
                           loss_wm_kl=loss_kl.mean(),
                           loss_wm_kl_max=loss_kl.max(),
                           entropy_prior=entropy_prior.mean(),
                           entropy_post=entropy_post.mean(),
                           )

            # Predictions from prior

            if image_pred is not None:
                logprob_img = self._decoder_image.loss(image_pred, image)
                logprob_img = -logavgexp(-logprob_img, dim=-1)  # This is *negative*-log-prob, so actually positive, same as loss
                log_tensors.update(logprob_img=logprob_img)
                metrics.update(logprob_img=logprob_img.mean())

            if reward_pred is not None:
                logprob_reward = self._decoder_reward.loss(reward_pred, reward)
                logprob_reward = -logavgexp(-logprob_reward, dim=-1)
                log_tensors.update(logprob_reward=logprob_reward)
                metrics.update(logprob_reward=logprob_reward.mean())

                reward_pos = (reward.select(-1, 0) > 0)  # mask where reward is *positive*
                logprob_reward_pos = (logprob_reward * reward_pos).sum() / reward_pos.sum()
                reward_neg = (reward.select(-1, 0) < 0)  # mask where reward is *negative*
                logprob_reward_neg = (logprob_reward * reward_neg).sum() / reward_neg.sum()
                metrics.update(logprob_reward_pos=logprob_reward_pos,
                               logprob_reward_neg=logprob_reward_neg)

            if terminal_pred is not None:
                logprob_terminal = self._decoder_terminal.loss(terminal_pred, terminal)
                logprob_terminal = -logavgexp(-logprob_terminal, dim=-1)
                log_tensors.update(logprob_terminal=logprob_terminal)
                metrics.update(logprob_terminal=logprob_terminal.mean())

                terminal_1 = (terminal.select(-1, 0) == 1)  # mask where terminal is 1
                logprob_terminal_1 = (logprob_terminal * terminal_1).sum() / terminal_1.sum()
                metrics.update(logprob_terminal_1=logprob_terminal_1)

        return loss_model.mean(), metrics, log_tensors


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
#                        loss_wm_image=loss_image.detach(),
#                        loss_wm=loss_image.detach(),
#                        loss_map=loss_map.detach(),
#                        #    **metrics_map
#                        )

#         return loss, metrics, log_tensors

class DirectHead(nn.Module):

    def __init__(self, decoder: ConvDecoder):
        super().__init__()
        self._decoder = decoder

    def forward(self, state, obs, do_image_pred=False):
        obs_pred = self._decoder.forward(state)
        return (obs_pred, )

    def loss(self, obs_pred: TensorNBICHW, obs_target: TensorNBICHW, map_coord: TensorNBI4):
        loss = self._decoder.loss(obs_pred, obs_target)  # (N,B,I)
        loss = -logavgexp(-loss, dim=-1)  # (N,B,I) => (N,B)
        with torch.no_grad():
            acc_map = self._decoder.accuracy(obs_pred, obs_target, map_coord)
            tensors = dict(loss_map=loss.detach(),
                           acc_map=acc_map)
            metrics = dict(loss_map=loss.mean(),
                           acc_map=nanmean(acc_map))
        return loss.mean(), metrics, tensors

    def to_distr(self, obs_pred):
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
             obs_target, map_coord,
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

    def to_distr(self, obs_rec, prior, post, obs_pred):
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

    def loss(self, obs_pred: TensorNBICHW, obs_target: TensorNBICHW, map_coord: TensorNBI4):
        return torch.square(self._dummy), {}, {}

    def to_distr(self, output):
        assert len(output.shape) == 6  # (N,B,I,C,H,W)
        x = output.mean(dim=2)  # (N,B,I,C,H,W) => (N,B,C,H,W)
        x = x.permute(0, 1, 3, 4, 2)  # (N,B,C,H,W) => (N,B,H,W,C)
        return D.Normal(x, torch.ones_like(x) / 255.0)
