import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch import Tensor

from .functions import *
from .common import *


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
                 actor_grad='reinforce',
                 actor_dist='onehot'
                 ):
        super().__init__()
        self.in_dim = in_dim
        self.out_actions = out_actions
        self.gamma = gamma
        self.lambda_ = lambda_gae
        self.entropy_weight = entropy_weight
        self.target_interval = target_interval
        self.actor_grad = actor_grad
        self.actor_dist = actor_dist

        actor_out_dim = out_actions if actor_dist == 'onehot' else 2 * out_actions
        self.actor = MLP(in_dim, actor_out_dim, hidden_dim, hidden_layers, layer_norm)
        self.critic = MLP(in_dim, 1, hidden_dim, hidden_layers, layer_norm)
        self.critic_target = MLP(in_dim, 1, hidden_dim, hidden_layers, layer_norm)
        self.critic_target.requires_grad_(False)
        self.train_steps = 0

    def forward_actor(self, features: Tensor) -> D.Distribution:
        y = self.actor.forward(features).float()  # .float() to force float32 on AMP

        if self.actor_dist == 'onehot':
            return D.OneHotCategorical(logits=y)
        
        if self.actor_dist == 'normal_tanh':
            return normal_tanh(y)

        if self.actor_dist == 'tanh_normal':
            return tanh_normal(y)

        assert False, self.actor_dist

    def forward_value(self, features: Tensor) -> Tensor:
        y = self.critic.forward(features)
        return y

    def training_step(self,
                      features: TensorJMF,
                      actions: TensorHMA,
                      rewards: D.Distribution,
                      terminals: D.Distribution,
                      log_only=False
                      ):
        if not log_only:
            if self.train_steps % self.target_interval == 0:
                self.update_critic_target()
            self.train_steps += 1

        reward1: TensorHM = rewards.mean[1:]
        terminal0: TensorHM = terminals.mean[:-1]
        terminal1: TensorHM = terminals.mean[1:]

        # GAE from https://arxiv.org/abs/1506.02438 eq (16)
        #   advantage_gae[t] = advantage[t] + (gamma lambda) advantage[t+1] + (gamma lambda)^2 advantage[t+2] + ...

        value_t: TensorJM = self.critic_target.forward(features)
        value0t: TensorHM = value_t[:-1]
        value1t: TensorHM = value_t[1:]
        advantage = - value0t + reward1 + self.gamma * (1.0 - terminal1) * value1t
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
        # Note: if lambda=0, then advantage_gae=advantage, then value_target = advantage + value0t = reward + gamma * value1t
        value_target = advantage_gae + value0t

        # When calculating losses, should ignore terminal states, or anything after, so:
        #   reality_weight[i] = (1-terminal[0]) (1-terminal[1]) ... (1-terminal[i])
        # Note this takes care of the case when initial state features[0] is terminal - it will get weighted by (1-terminals[0]).
        reality_weight = (1 - terminal0.detach()).log().cumsum(dim=0).exp()

        # Critic loss

        value: TensorJM = self.critic.forward(features.detach())
        value0: TensorHM = value[:-1]
        loss_critic = 0.5 * torch.square(value_target.detach() - value0)
        loss_critic = (loss_critic * reality_weight).mean()

        # Actor loss

        policy_distr = self.forward_actor(features.detach()[:-1])  # TODO: we could reuse this from dream()
        if self.actor_grad == 'reinforce':
            action_logprob = policy_distr.log_prob(actions.detach())
            loss_policy = - action_logprob * advantage_gae.detach()
        elif self.actor_grad == 'dynamics':
            loss_policy = - value_target
        else:
            assert False, self.actor_grad

        policy_entropy = policy_distr.entropy()
        loss_actor = loss_policy - self.entropy_weight * policy_entropy
        loss_actor = (loss_actor * reality_weight).mean()
        assert (loss_policy.requires_grad and policy_entropy.requires_grad) or not loss_critic.requires_grad

        with torch.no_grad():
            metrics = dict(loss_critic=loss_critic.detach(),
                           loss_actor=loss_actor.detach(),
                           policy_entropy=policy_entropy.mean(),
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
                           )

        return (loss_actor, loss_critic), metrics, tensors

    def update_critic_target(self):
        self.critic_target.load_state_dict(self.critic.state_dict())  # type: ignore
