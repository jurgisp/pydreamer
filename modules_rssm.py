from typing import Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch import Tensor

from modules_tools import *


class RSSMCore(nn.Module):

    def __init__(self, embed_dim=256, action_dim=7, deter_dim=200, stoch_dim=30, hidden_dim=200, global_dim=30, gru_layers=1):
        super().__init__()
        self._cell = RSSMCell(embed_dim, action_dim, deter_dim, stoch_dim, hidden_dim, global_dim, gru_layers)

    def forward(self,
                embed: Tensor,       # tensor(N, B, E)
                action: Tensor,      # tensor(N, B, A)
                reset: Tensor,       # tensor(N, B)
                in_state: Tuple[Tensor, Tensor],    # [(BI,D) (BI,S)]
                glob_state: Any,     # (B,G)?
                I: int = 1,
                imagine=False,       # If True, will imagine sequence, not using observations to form posterior
                ):

        n, b = embed.shape[:2]

        # Multiply batch dimension by I samples

        def expand(x):
            # (N,B,X) -> (N,BI,X)
            return x.unsqueeze(2).expand(n, b, I, -1).reshape(n, b * I, -1)

        embeds = expand(embed).unbind(0)     # (N,B,...) => List[(BI,...)]
        actions = expand(action).unbind(0)
        reset_masks = expand(~reset.unsqueeze(2)).unbind(0)
        noises = torch.normal(torch.zeros((n, b, I, self._cell._stoch_dim), device=embed.device),
                              torch.ones((n, b, I, self._cell._stoch_dim), device=embed.device))
        noises = noises.reshape(n, b * I, -1).unbind(0)  # List[(BI,S)]

        priors = []
        posts = []
        states_h = []
        samples = []
        (h, z) = in_state

        for i in range(n):
            if not imagine:
                post, (h, z) = self._cell.forward(embeds[i], actions[i], reset_masks[i], (h, z), glob_state, noises[i])
            else:
                post, (h, z) = self._cell.forward_prior(actions[i], (h, z), glob_state, noises[i])  # post=prior in this case
            posts.append(post)
            states_h.append(h)
            samples.append(z)

        posts = torch.stack(posts)          # (N,BI,2S)
        states_h = torch.stack(states_h)    # (N,BI,D)
        samples = torch.stack(samples)      # (N,BI,S)
        priors = self._cell.batch_prior(states_h)  # (N,BI,2S)
        features = self.to_feature(states_h, samples)   # (N,BI,D+S)

        posts = posts.reshape(n, b, I, -1)  # (N,BI,X) => (N,B,I,X)
        states_h = states_h.reshape(n, b, I, -1)
        samples = samples.reshape(n, b, I, -1)
        priors = priors.reshape(n, b, I, -1)
        features = features.reshape(n, b, I, -1)

        return (
            priors,                      # tensor(N,B,I,2S)
            posts,                       # tensor(N,B,I,2S)
            samples,                     # tensor(N,B,I,S)
            features,                    # tensor(N,B,I,D+S)
            (h.detach(), z.detach()),
        )

    def init_state(self, batch_size):
        return self._cell.init_state(batch_size)

    def to_feature(self, h: Tensor, z: Tensor) -> Tensor:
        return torch.cat((h, z), -1)

    def feature_replace_z(self, features: Tensor, z: Tensor):
        h, _ = features.split([self._cell._deter_dim, self._cell._stoch_dim], -1)
        return self.to_feature(h, z)


class RSSMCell(nn.Module):

    def __init__(self, embed_dim, action_dim, deter_dim, stoch_dim, hidden_dim, global_dim, gru_layers=1):
        super().__init__()
        self._stoch_dim = stoch_dim
        self._deter_dim = deter_dim
        self._global_dim = global_dim

        self._z_mlp = nn.Linear(stoch_dim, hidden_dim)
        self._a_mlp = nn.Linear(action_dim, hidden_dim, bias=False)  # No bias, because outputs are added
        # self._g_mlp = nn.Linear(global_dim, hidden_dim, bias=False)  # TODO

        self._gru = nn.GRUCell(hidden_dim, deter_dim // gru_layers)
        self._gru_layers = gru_layers
        if gru_layers > 1:
            assert gru_layers == 2
            self._gru2 = nn.GRUCell(deter_dim // gru_layers, deter_dim // gru_layers)

        self._prior_mlp_h = nn.Linear(deter_dim, hidden_dim)
        # self._prior_mlp_g = nn.Linear(global_dim, hidden_dim, bias=False)  # TODO
        self._prior_mlp = nn.Linear(hidden_dim, 2 * stoch_dim)

        self._post_mlp_h = nn.Linear(deter_dim, hidden_dim)
        # self._post_mlp_g = nn.Linear(global_dim, hidden_dim, bias=False)  # TODO
        self._post_mlp_e = nn.Linear(embed_dim, hidden_dim, bias=False)
        self._post_mlp = nn.Linear(hidden_dim, 2 * stoch_dim)

    def init_state(self, batch_size):
        device = next(self._gru.parameters()).device
        return (
            torch.zeros((batch_size, self._deter_dim), device=device),
            torch.zeros((batch_size, self._stoch_dim), device=device),
        )

    def forward(self,
                embed: Tensor,                    # tensor(B,E)
                action: Tensor,                   # tensor(B,A)
                reset_mask: Tensor,               # tensor(B,1)
                in_state: Tuple[Tensor, Tensor],  # tensor(B,D+S)
                glob_state: Tensor,               # tensor(B,G)
                noise: Tensor,                    # tensor(B,S)
                ) -> Tuple[Tensor,
                           Tuple[Tensor, Tensor]]:

        in_h, in_z = in_state
        in_h = in_h * reset_mask
        in_z = in_z * reset_mask

        za = F.elu(self._z_mlp(in_z) + self._a_mlp(action))    # (B, H)
        if self._gru_layers == 1:
            h = self._gru(za, in_h)                                             # (B, D)
        else:
            in_h1, in_h2 = in_h.chunk(2, -1)
            h1 = self._gru(za, in_h1)
            h2 = self._gru2(h1, in_h2)
            h = torch.cat((h1, h2), -1)

        post_in = F.elu(self._post_mlp_h(h) + self._post_mlp_e(embed))
        post = self._post_mlp(post_in)                                    # (B, 2*S)
        sample = rsample(post, noise)                                     # (B, S)

        return (
            post,                         # tensor(B, 2*S)
            (h, sample),                  # tensor(B, D+S+G)
        )

    def forward_prior(self,
                      action: Tensor,                   # tensor(B,A)
                      in_state: Tuple[Tensor, Tensor],  # tensor(B,D+S)
                      glob_state: Tensor,               # tensor(B,G)
                      noise: Tensor,                    # tensor(B,S)
                      ) -> Tuple[Tensor,
                                 Tuple[Tensor, Tensor]]:

        in_h, in_z = in_state

        za = F.elu(self._z_mlp(in_z) + self._a_mlp(action))    # (B,H)
        if self._gru_layers == 1:
            h = self._gru(za, in_h)                                             # (B, D)
        else:
            in_h1, in_h2 = in_h.chunk(2, -1)
            h1 = self._gru(za, in_h1)
            h2 = self._gru2(h1, in_h2)
            h = torch.cat((h1, h2), -1)

        prior = self._prior_mlp(F.elu(self._prior_mlp_h(h)))   # (B,2S)
        sample = rsample(prior, noise)                         # (B,S)

        return (
            prior,                        # (B,2S)
            (h, sample),                  # (B,D+S)
        )

    def batch_prior(self,
                    h: Tensor,     # tensor(N, B, D)
                    ) -> Tensor:
        return self._prior_mlp(F.elu(self._prior_mlp_h(h)))  # tensor(B,2S)
