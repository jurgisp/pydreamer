import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch import Tensor

from models.functions import *
from models.common import *


class ActorCritic(nn.Module):

    def __init__(self,
                 in_dim,
                 out_actions,
                 hidden_dim=400,
                 hidden_layers=4,
                 layer_norm=True,
                 gamma=0.999,
                 lambda_gae=0.95,
                 entropy_weight=1e-3,
                 target_interval=100,
                 ):
        super().__init__()
        self.in_dim = in_dim
        self.out_actions = out_actions
        self.gamma = gamma
        self.lambda_ = lambda_gae
        self.entropy_weight = entropy_weight
        self.target_interval = target_interval
        self.actor = MLP(in_dim, out_actions, hidden_dim, hidden_layers, layer_norm)
        self.critic = MLP(in_dim, 1, hidden_dim, hidden_layers, layer_norm)
        self.critic_target = MLP(in_dim, 1, hidden_dim, hidden_layers, layer_norm)
        self.train_steps = 0

    def forward_actor(self, features: Tensor) -> Tensor:
        # Would be nice to return D.OneHotCategorical here, but there's a potential problem with AMP:
        # D.Categorical(logits=y).logits remains float16, because logsumexp() doesn't autocast to float32,
        # whereas y.log_softmax(-1) does autocast
        # TODO: D.OneHotCategorical(logits=y.float()) seems to work fine to force float32
        y = self.actor.forward(features)
        logits = y.log_softmax(-1)
        return logits

    def forward_value(self, features: Tensor) -> Tensor:
        y = self.critic.forward(features)
        return y

    def training_step(self,
                      features: TensorJMF,
                      rewards: D.Distribution,
                      terminals: D.Distribution,
                      actions: TensorHMA,
                      log_only=False
                      ):
        if not log_only:
            if self.train_steps % self.target_interval == 0:
                self.update_critic_target()
            self.train_steps += 1

        reward1: TensorHM = rewards.mean[1:]
        terminal0: TensorHM = terminals.mean[:-1]
        terminal1: TensorHM = terminals.mean[1:]
        policy_logits = self.forward_actor(features[:-1])
        value: TensorJM = self.critic.forward(features)
        value0: TensorHM = value[:-1]
        with torch.no_grad():
            value_t: TensorJM = self.critic_target.forward(features)
            value0t: TensorHM = value_t[:-1]
            value1t: TensorHM = value_t[1:]
            advantage = - value0t + reward1 + self.gamma * (1.0 - terminal1) * value1t
            assert not advantage.requires_grad

        # GAE from https://arxiv.org/abs/1506.02438 eq (16)
        #   advantage_gae[t] = advantage[t] + (gamma lambda) advantage[t+1] + (gamma lambda)^2 advantage[t+2] + ...
        advantage_gae = []
        agae = None
        for adv, term in zip(reversed(advantage.unbind()), reversed(terminal1.unbind())):
            if agae is None:
                agae = adv
            else:
                agae = adv + self.lambda_ * self.gamma * (1.0 - term) * agae
            advantage_gae.append(agae)
        advantage_gae.reverse()
        advantage_gae = torch.stack(advantage_gae)
        assert not advantage_gae.requires_grad

        # Sanity check #1: if lambda=0, then advantage_gae=advantage, then
        #   value_target = advantage + value0t
        #                = reward + gamma * value1t
        value_target = advantage_gae + value0t

        # Sanity check #2: if lambda=1 then
        #   advantage_gae[t] = value_target_mc[t]
        #                    = reward[t] + g[t] reward[t+1] + g[t]g[t+1] reward[t+2] + ... + g[t]g[t+1]g[t+2]g[..] value1t[-1]
        # where
        #   g[t] = gamma * (1 - terminal[t])
        # is the adjusted discount

        # if self.lambda_ == 1:
        #     value_target_mc = []
        #     v = value1t[-1]
        #     for rew, term in zip(reversed(reward.unbind()), reversed(terminal1.unbind())):
        #         v = rew + self.gamma * (1.0 - term) * v
        #         value_target_mc.append(v)
        #     value_target_mc.reverse()
        #     value_target_mc = torch.stack(value_target_mc)
        #     assert torch.allclose(value_target, value_target_mc)

        # When calculating losses, should ignore terminal states, or anything after, so:
        #   reality_weight[i] = (1-terminal[0]) (1-terminal[1]) ... (1-terminal[i])
        # Note this takes care of the case when initial state features[0] is terminal - it will get weighted by (1-terminals[0]).
        reality_weight = (1 - terminal0).log().cumsum(dim=0).exp()

        loss_value = 0.5 * torch.square(value_target - value0)
        action_logprob = (policy_logits * actions).sum(-1)
        loss_policy = - action_logprob * advantage_gae
        policy_entropy = - (policy_logits * policy_logits.exp()).sum(-1)
        assert (loss_policy.requires_grad and policy_entropy.requires_grad) or not loss_value.requires_grad

        loss_value = (loss_value * reality_weight).mean()
        loss_policy = (loss_policy * reality_weight).mean()
        policy_entropy = (policy_entropy * reality_weight).mean()

        loss_critic = loss_value
        loss_actor = loss_policy - self.entropy_weight * policy_entropy

        with torch.no_grad():
            metrics = dict(loss_critic=loss_critic.detach(),
                           loss_actor=loss_actor.detach(),
                           policy_entropy=policy_entropy.detach(),
                           policy_value=value0[0].mean(),  # Value of real states
                           policy_value_im=value0.mean(),  # Value of imagined states
                           policy_reward=reward1.mean(),
                           policy_reward_std=reward1.std(),
                           )
            tensors = dict(value=value.detach(),
                           value_target=value_target.detach(),
                           value_advantage=advantage.detach(),
                           value_advantage_gae=advantage_gae.detach(),
                           value_weight=reality_weight.detach(),
                           action_prob=action_logprob.exp().detach(),
                           )

        return (loss_actor, loss_critic), metrics, tensors

    def update_critic_target(self):
        self.critic_target.load_state_dict(self.critic.state_dict())  # type: ignore
