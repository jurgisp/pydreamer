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
from models.heads import *


class Dreamer(TrainableModel):

    def __init__(self, conf):
        super().__init__()

        state_dim = conf.deter_dim + conf.stoch_dim * (conf.stoch_discrete or 1) + conf.global_dim
        n_map_coords = 4
        map_state_dim = state_dim + n_map_coords

        # Map decoder

        if conf.map_model == 'vae':
            if conf.map_decoder == 'cnn':
                map_model = VAEHead(
                    encoder=ConvEncoder(in_channels=conf.map_channels),
                    decoder=ConvDecoder(in_dim=map_state_dim + conf.map_stoch_dim,
                                        mlp_layers=2,
                                        layer_norm=conf.layer_norm,
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
                                        layer_norm=conf.layer_norm,
                                        out_channels=conf.map_channels))

            else:
                map_model = DirectHead(
                    decoder=DenseDecoder(in_dim=map_state_dim,
                                         out_shape=(conf.map_channels, conf.map_size, conf.map_size),
                                         hidden_dim=conf.map_hidden_dim,
                                         hidden_layers=conf.map_hidden_layers,
                                         layer_norm=conf.layer_norm))    # type: ignore
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
            stoch_discrete=conf.stoch_discrete,
            hidden_dim=conf.hidden_dim,
            kl_weight=conf.kl_weight,
            image_weight=conf.image_weight,
            vecobs_weight=conf.vecobs_weight,
            reward_weight=conf.reward_weight,
            terminal_weight=conf.terminal_weight,
            embed_rnn=conf.embed_rnn != 'none',
            gru_layers=conf.gru_layers,
            gru_type=conf.gru_type,
            kl_balance=conf.kl_balance,
        )

        # Actor critic

        self.ac = ActorCritic(in_dim=state_dim,
                              out_actions=conf.action_dim,
                              layer_norm=conf.layer_norm,
                              gamma=conf.gamma,
                              lambda_gae=conf.lambda_gae,
                              entropy_weight=conf.entropy,
                              target_interval=conf.target_interval,
                              )

    @property
    def submodels(self):
        return (self.wm._encoder, self.wm._decoder_image, self.wm._core, self.wm._input_rnn, self.map_model)

    def optimizers(self, conf):
        optimizer_wm = torch.optim.AdamW(self.wm.parameters(), lr=conf.adam_lr, eps=conf.adam_eps)  # type: ignore
        optimizer_map = torch.optim.AdamW(self.map_model.parameters(), lr=conf.adam_lr, eps=conf.adam_eps)  # type: ignore
        optimizer_actor = torch.optim.AdamW(self.ac._actor.parameters(), lr=conf.adam_lr_actor, eps=conf.adam_eps)  # type: ignore
        optimizer_critic = torch.optim.AdamW(self.ac._critic.parameters(), lr=conf.adam_lr_critic, eps=conf.adam_eps)  # type: ignore
        return optimizer_wm, optimizer_map, optimizer_actor, optimizer_critic

    def grad_clip(self, conf):
        grad_metrics = {
            'grad_norm': nn.utils.clip_grad_norm_(self.wm.parameters(), conf.grad_clip),
            'grad_norm_map': nn.utils.clip_grad_norm_(self.map_model.parameters(), conf.grad_clip),
            'grad_norm_actor': nn.utils.clip_grad_norm_(self.ac._actor.parameters(), conf.grad_clip_ac),
            'grad_norm_critic': nn.utils.clip_grad_norm_(self.ac._critic.parameters(), conf.grad_clip_ac),
        }
        return grad_metrics

    def init_state(self, batch_size: int):
        return self.wm.init_state(batch_size)

    def forward(self,
                image: TensorNBCHW,   # (1,B,C,H,W)
                vecobs: Tensor,       # (1,V)
                prev_reward: Tensor,  # (1,B)
                prev_action: Tensor,  # (1,B,A)
                reset: Tensor,        # (1,B)
                in_state: Any,
                ):
        N, B = image.shape[:2]
        assert N == 1

        # Forward (world model)

        terminal = torch.zeros_like(reset)
        output = self.wm.forward(image, vecobs, prev_reward, terminal, prev_action, reset, in_state)
        features = output[0]
        out_state = output[-1]

        # Forward (actor critic)

        feature = features[:, :, 0]  # (N=1,B,I=1,F) => (1,B,F)
        action_logits = self.ac.forward_actor(feature)  # (1,B,A)
        value = self.ac.forward_value(feature)  # (1,B)

        return action_logits, value, out_state

    def training_step(self,
                      image: TensorNBCHW,
                      vecobs: Tensor,
                      reward: Tensor,
                      terminal: Tensor,
                      action: Tensor,     # (N,B,A)
                      reset: Tensor,      # (N,B)
                      map: Tensor,
                      map_coord: Tensor,  # (N,B,4)
                      map_seen_mask: Tensor,
                      in_state: Any,
                      I: int = 1,         # IWAE samples
                      H: int = 1,        # Imagination horizon
                      imagine_dropout=0,      # If True, will imagine sequence, not using observations to form posterior
                      do_image_pred=False,
                      do_output_tensors=False,
                      do_dream_tensors=False,
                      ):
        N, B = image.shape[:2]

        # Forward (world model)

        output = self.wm.forward(image, vecobs, reward, terminal, action, reset, in_state, I=I, imagine_dropout=imagine_dropout, do_image_pred=do_image_pred)
        features = output[0]
        states = output[-2]
        out_state = output[-1]

        # Forward (map)

        map_coord = map_coord.unsqueeze(2).expand(N, B, I, -1)  # TODO: move inside map_model.forward()
        map = map.unsqueeze(2).expand(N, B, I, *map.shape[2:])
        map_features = torch.cat((features, map_coord), dim=-1)
        map_features = map_features.detach()
        map_out = self.map_model.forward(map_features, map, do_image_pred=do_image_pred)

        # Forward (dream)

        in_state_dream: StateB = map_structure(states, lambda x: flatten_batch(x.detach())[0])  # type: ignore  # (N,B,I) => (NBI)
        features_dream, actions_dream = self.dream(in_state_dream, H)   # (H+1,NBI,D) - features_dream includes the starting "real" features at features_dream[0]
        features_dream = features_dream.detach()  # Not using dynamics gradients for now, just Reinforce
        with torch.no_grad():  # careful not to invoke modules first time under no_grad (https://github.com/pytorch/pytorch/issues/60164)
            rewards_dream = self.wm._decoder_reward.forward(features_dream)      # (H+1,NBI)
            terminals_dream = self.wm._decoder_terminal.forward(features_dream)  # (H+1,NBI)

        # Loss

        loss_model, metrics, loss_tensors = self.wm.loss(*output, image, vecobs, reward, terminal, reset)
        metrics.update(loss=metrics['loss_wm'])

        loss_map, metrics_map, loss_tensors_map = self.map_model.loss(*map_out, map, map_coord, map_seen_mask)    # type: ignore
        metrics.update(**metrics_map)
        loss_tensors.update(**loss_tensors_map)

        losses_ac, metrics_ac, loss_tensors_ac = self.ac.training_step(features_dream, rewards_dream, terminals_dream, actions_dream)
        metrics.update(**metrics_ac)
        loss_tensors.update(policy_value=unflatten_batch(loss_tensors_ac['value'][0], (N, B, I)).mean(-1))

        # Predict

        out_tensors = {}
        if do_output_tensors:
            with torch.no_grad():
                image_pred, image_rec = self.wm.predict(*output)
                map_rec = self.map_model.predict(*map_out)
                out_tensors = dict(image_rec=image_rec, map_rec=map_rec)
                if image_pred is not None:
                    out_tensors.update(image_pred=image_pred)

        # Dream for a log sample.

        dream_tensors = {}
        if do_dream_tensors:
            with torch.no_grad():  # careful not to invoke modules first time under no_grad (https://github.com/pytorch/pytorch/issues/60164)
                # The reason we don't just take real features_dream is because it's really big (H*N*B*I),
                # and here for inspection purposes we only dream from first step, so it's (H*B).
                # Oh, and we set here H=N-1, so we get (N,B), and the dreamed experience aligns with actual.
                in_state_dream: StateB = map_structure(states, lambda x: x.detach()[0, :, 0])  # type: ignore  # (N,B,I) => (B)
                features_dream, actions_dream = self.dream(in_state_dream, N - 1)      # H = N-1
                rewards_dream = self.wm._decoder_reward.forward(features_dream)      # (H+1,B) = (N,B)
                terminals_dream = self.wm._decoder_terminal.forward(features_dream)  # (H+1,B) = (N,B)
                image_dream = self.wm._decoder_image.forward(features_dream)
                _, _, loss_tensors_ac = self.ac.training_step(features_dream, rewards_dream, terminals_dream, actions_dream, log_only=True)
                # The tensors are intentionally named same as in out_tensors, so the logged npz looks the same for dreamed or not
                dream_tensors = dict(action_pred=torch.cat([action[:1], actions_dream]),  # first action is real from previous step
                                     reward_pred=rewards_dream.mean,  # reward_pred is also set in loss_tensors, if do_image_pred==True
                                     terminal_pred=terminals_dream.mean,
                                     image_pred=image_dream,
                                     **loss_tensors_ac)
                assert dream_tensors['action_pred'].shape == action.shape
                assert dream_tensors['image_pred'].shape == image.shape

        losses = (loss_model, loss_map, *losses_ac)
        return losses, metrics, loss_tensors, out_state, out_tensors, dream_tensors

    def dream(self, in_state: StateB, H: int):
        NBI = len(in_state[0])  # Imagine batch size = N*B*I
        noises = self.wm._core.generate_noises(H, (NBI, ), in_state[0].device)
        features = []
        actions = []
        state = in_state

        for i in range(H):
            feature = self.wm._core.to_feature(*state)
            action = D.OneHotCategorical(logits=self.ac.forward_actor(feature)).sample()
            features.append(feature)
            actions.append(action)
            _, state = self.wm._core._cell.forward_prior(action, None, state, noises[i])

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
                 stoch_discrete=0,
                 hidden_dim=200,
                 kl_weight=1.0,
                 image_weight=1.0,
                 vecobs_weight=1.0,
                 reward_weight=1.0,
                 terminal_weight=1.0,
                 embed_rnn=False,
                 embed_rnn_dim=512,
                 gru_layers=1,
                 gru_type='gru',
                 kl_balance=0.5,
                 ):
        super().__init__()
        self._deter_dim = deter_dim
        self._stoch_dim = stoch_dim
        self._stoch_discrete = stoch_discrete
        self._global_dim = 0
        self._kl_weight = kl_weight
        self._image_weight = image_weight
        self._vecobs_weight = vecobs_weight
        self._reward_weight = reward_weight
        self._terminal_weight = terminal_weight
        self._embed_rnn = embed_rnn
        self._kl_balance = None if kl_balance == 0.5 else kl_balance

        # Encoder

        self._reward_input = conf.reward_input
        if conf.reward_input:
            encoder_channels = conf.image_channels + 2  # + reward, terminal
        else:
            encoder_channels = conf.image_channels

        if conf.image_encoder == 'cnn':
            self._encoder = ConvEncoder(in_channels=encoder_channels,
                                        cnn_depth=conf.cnn_depth)
        else:
            self._encoder = DenseEncoder(in_dim=conf.image_size * conf.image_size * encoder_channels,
                                         out_dim=256,
                                         hidden_layers=conf.image_encoder_layers,
                                         layer_norm=conf.layer_norm)

        if self._embed_rnn:
            self._input_rnn = GRU2Inputs(input1_dim=self._encoder.out_dim,
                                         input2_dim=action_dim,
                                         mlp_dim=embed_rnn_dim,
                                         state_dim=embed_rnn_dim,
                                         bidirectional=True)
        else:
            self._input_rnn = None

        self._encoder_vecobs = MLP(64, 256, hidden_dim=400, hidden_layers=2, layer_norm=conf.layer_norm)

        # Decoders

        state_dim = conf.deter_dim + conf.stoch_dim * (conf.stoch_discrete or 1) + conf.global_dim
        if conf.image_decoder == 'cnn':
            self._decoder_image = ConvDecoder(in_dim=state_dim,
                                              out_channels=conf.image_channels,
                                              cnn_depth=conf.cnn_depth)
        else:
            self._decoder_image = DenseDecoder(in_dim=state_dim,
                                               out_shape=(conf.image_channels, conf.image_size, conf.image_size),
                                               hidden_layers=conf.image_decoder_layers,
                                               layer_norm=conf.layer_norm,
                                               min_prob=conf.image_decoder_min_prob)

        if conf.reward_decoder_categorical:
            self._decoder_reward = DenseCategoricalSupportHead(in_dim=state_dim,
                                                               support=conf.reward_decoder_categorical,
                                                               hidden_layers=conf.reward_decoder_layers,
                                                               layer_norm=conf.layer_norm)
        else:
            self._decoder_reward = DenseNormalHead(in_dim=state_dim, hidden_layers=conf.reward_decoder_layers, layer_norm=conf.layer_norm)
        self._decoder_terminal = DenseBernoulliHead(in_dim=state_dim, hidden_layers=conf.terminal_decoder_layers, layer_norm=conf.layer_norm)
        self._decoder_vecobs = DenseNormalHead(in_dim=state_dim, out_dim=64, hidden_layers=4, layer_norm=conf.layer_norm)

        # RSSM

        self._core = RSSMCore(embed_dim=self._encoder.out_dim + 256 + (2 * embed_rnn_dim if embed_rnn else 0),
                              action_dim=action_dim,
                              deter_dim=deter_dim,
                              stoch_dim=stoch_dim,
                              stoch_discrete=stoch_discrete,
                              hidden_dim=hidden_dim,
                              global_dim=self._global_dim,
                              gru_layers=gru_layers,
                              gru_type=gru_type,
                              layer_norm=conf.layer_norm)

        # Init

        for m in self.modules():
            init_weights_tf2(m)

    def init_state(self, batch_size: int) -> Tuple[Any, Any]:
        return self._core.init_state(batch_size)

    def forward(self,
                image: TensorNBCHW,
                vecobs: Tensor,
                reward: TensorNB,
                terminal: TensorNB,
                action: Tensor,    # tensor(N, B, A)
                reset: Tensor,     # tensor(N, B)
                in_state: Any,
                I: int = 1,
                imagine_dropout=0,     # If 1, will imagine sequence, not using observations to form posterior
                do_image_pred=False,
                ):

        N, B, C, H, W = image.shape
        noises = self._core.generate_noises(N, (B * I, ), image.device)  # Belongs to RSSM but need to do here for perf

        # Encoder

        if self._reward_input:
            reward_plane = reward.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand((N, B, 1, H, W))
            terminal_plane = terminal.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand((N, B, 1, H, W))
            observation = torch.cat([image,  # (N,B,C+2,H,W)
                                     reward_plane.to(image.dtype),
                                     terminal_plane.to(image.dtype)], dim=-3)
        else:
            observation = image

        embed = self._encoder.forward(observation)  # (N,B,E)
        embed_vecobs = self._encoder_vecobs(vecobs)
        embed = torch.cat((embed, embed_vecobs), dim=-1)  # (N,B,E+256)
        if self._input_rnn:
            embed_rnn, _ = self._input_rnn.forward(embed, action)  # (N,B,2E)
            embed = torch.cat((embed, embed_rnn), dim=-1)  # (N,B,3E+256)

        # RSSM

        prior, post, post_samples, features, states, out_state = self._core.forward(embed, action, reset, in_state, None, noises, I=I, imagine_dropout=imagine_dropout)

        # Decoder

        image_rec = self._decoder_image.forward(features)
        vecobs_rec = self._decoder_vecobs.forward(features)
        reward_rec = self._decoder_reward.forward(features)
        terminal_rec = self._decoder_terminal.forward(features)

        # Predictions

        image_pred, reward_pred, terminal_pred = None, None, None
        if do_image_pred:
            prior_samples = self._core.zdistr(prior).sample().reshape(post_samples.shape)
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
            vecobs_rec,
            reward_rec,
            terminal_rec,
            image_pred,                  # Optional[(N,B,I,C,H,W)]
            reward_pred,
            terminal_pred,
            states,
            out_state,
        )

    def predict(self,
                features, prior, post, post_samples, image_rec, vecobs_rec, reward_rec, terminal_rec, image_pred, reward_pred, terminal_pred, states, out_state,     # forward() output
                ) -> Tuple[Tensor, Tensor]:
        # TODO: obsolete this method
        if image_pred is not None:
            image_pred = image_pred.mean(dim=2)
        image_rec = image_rec.mean(dim=2)  # (N,B,I,C.H,W) => (N,B,C.H,W)
        return (image_pred, image_rec)

    def loss(self,
             features, prior, post, post_samples, image_rec, vecobs_rec, reward_rec, terminal_rec, image_pred, reward_pred, terminal_pred, states, out_state,     # forward() output
             image, vecobs, reward, terminal,
             reset,
             ):
        N, B, I = image_rec.shape[:3]
        image = image.unsqueeze(2).expand(N, B, I, *image.shape[2:])
        vecobs = vecobs.unsqueeze(2).expand(N, B, I, *vecobs.shape[2:])
        reward = reward.unsqueeze(2).expand(N, B, I, *reward.shape[2:])
        terminal = terminal.unsqueeze(2).expand(N, B, I, *terminal.shape[2:])

        # Reconstruction

        loss_image = self._decoder_image.loss(image_rec, image)  # (N,B,I)
        loss_vecobs = self._decoder_vecobs.loss(vecobs_rec, vecobs)  # (N,B,I)
        loss_reward = self._decoder_reward.loss(reward_rec, reward)  # (N,B,I)
        loss_terminal = self._decoder_terminal.loss(terminal_rec, terminal)  # (N,B,I)
        assert (loss_image.requires_grad and loss_reward.requires_grad and loss_terminal.requires_grad) or not features.requires_grad

        # KL

        d = self._core.zdistr
        dprior = d(prior)
        dpost = d(post)
        loss_kl_exact = D.kl.kl_divergence(dpost, dprior)  # (N,B,I)
        if I == 1:
            # Analytic KL loss, standard for VAE
            if not self._kl_balance:
                loss_kl = loss_kl_exact
            else:
                loss_kl_postgrad = D.kl.kl_divergence(dpost, d(prior.detach()))
                loss_kl_priograd = D.kl.kl_divergence(d(post.detach()), dprior)
                loss_kl = (1 - self._kl_balance) * loss_kl_postgrad + self._kl_balance * loss_kl_priograd
        else:
            # Sampled KL loss, for IWAE
            z = post_samples.reshape(dpost.batch_shape + dpost.event_shape)
            loss_kl = dpost.log_prob(z) - dprior.log_prob(z)

        # Total loss

        assert loss_kl.shape == loss_image.shape == loss_vecobs.shape == loss_reward.shape == loss_terminal.shape
        loss_model = -logavgexp(-(
            self._kl_weight * loss_kl
            + self._image_weight * loss_image
            + self._vecobs_weight * loss_vecobs
            + self._reward_weight * loss_reward
            + self._terminal_weight * loss_terminal
        ), dim=-1)

        # IWAE according to paper

        # with torch.no_grad():
        #     weights = F.softmax(-(loss_image + loss_kl), dim=-1)
        # dloss = (weights * (loss_image + loss_kl)).sum(dim=-1).mean()

        # Metrics

        with torch.no_grad():
            loss_image = -logavgexp(-loss_image, dim=-1)
            loss_vecobs = -logavgexp(-loss_vecobs, dim=-1)
            loss_reward = -logavgexp(-loss_reward, dim=-1)
            loss_terminal = -logavgexp(-loss_terminal, dim=-1)
            loss_kl = -logavgexp(-loss_kl_exact, dim=-1)  # Log exact KL loss even when using IWAE, it avoids random negative values
            entropy_prior = dprior.entropy().mean(dim=-1)
            entropy_post = dpost.entropy().mean(dim=-1)

            log_tensors = dict(loss_kl=loss_kl.detach(),
                               loss_image=loss_image.detach(),
                               entropy_prior=entropy_prior,
                               entropy_post=entropy_post,
                               )

            metrics = dict(loss_wm=loss_model.mean(),
                           loss_wm_image=loss_image.mean(),
                           loss_wm_image_max=loss_image.max(),
                           loss_wm_vecobs=loss_vecobs.mean(),
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
                reward_pos = (reward.select(-1, 0) > 0)  # mask where reward is *positive*
                logprob_reward_pos = (logprob_reward * reward_pos).sum() / reward_pos.sum()
                reward_neg = (reward.select(-1, 0) < 0)  # mask where reward is *negative*
                logprob_reward_neg = (logprob_reward * reward_neg).sum() / reward_neg.sum()
                metrics.update(logprob_reward=logprob_reward.mean(),
                               logprob_reward_pos=logprob_reward_pos,
                               logprob_reward_neg=logprob_reward_neg)
                log_tensors.update(reward_pred=reward_pred.mean.mean(dim=-1))  # not quite loss tensor, but fine

            if terminal_pred is not None:
                logprob_terminal = self._decoder_terminal.loss(terminal_pred, terminal)
                logprob_terminal = -logavgexp(-logprob_terminal, dim=-1)
                terminal_1 = (terminal.select(-1, 0) == 1)  # mask where terminal is 1
                logprob_terminal_1 = (logprob_terminal * terminal_1).sum() / terminal_1.sum()
                metrics.update(logprob_terminal=logprob_terminal.mean(),
                               logprob_terminal_1=logprob_terminal_1)
                log_tensors.update(terminal_pred=terminal_pred.mean.mean(dim=-1))  # not quite loss tensor, but fine

        return loss_model.mean(), metrics, log_tensors
