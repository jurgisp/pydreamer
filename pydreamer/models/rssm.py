from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import models.rnn as my
from models.functions import *
from models.common import *


class RSSMCore(nn.Module):

    def __init__(self, embed_dim, action_dim, deter_dim, stoch_dim, stoch_discrete, hidden_dim, global_dim, gru_layers, gru_type, layer_norm):
        super().__init__()
        if stoch_discrete:
            self.cell = RSSMCellDiscrete(embed_dim, action_dim, deter_dim, stoch_dim, stoch_discrete, hidden_dim, global_dim, gru_layers, gru_type, layer_norm)
        else:
            self.cell = RSSMCell(embed_dim, action_dim, deter_dim, stoch_dim, hidden_dim, global_dim, gru_layers, gru_type, layer_norm)

    def forward(self,
                embed: Tensor,       # tensor(N, B, E)
                action: Tensor,      # tensor(N, B, A)
                reset: Tensor,       # tensor(N, B)
                in_state: Tuple[Tensor, Tensor],    # [(BI,D) (BI,S)]
                glob_state: Any,
                noises: List[Tensor],      # (N,B,I)
                I: int = 1,
                imagine_dropout=0,       # If 1, will imagine sequence, not using observations to form posterior
                ):

        n, b = embed.shape[:2]

        # Multiply batch dimension by I samples

        def expand(x):
            # (N,B,X) -> (N,BI,X)
            return x.unsqueeze(2).expand(n, b, I, -1).reshape(n, b * I, -1)

        embeds = expand(embed).unbind(0)     # (N,B,...) => List[(BI,...)]
        actions = expand(action).unbind(0)
        reset_masks = expand(~reset.unsqueeze(2)).unbind(0)

        priors = []
        posts = []
        states_h = []
        samples = []
        (h, z) = in_state

        for i in range(n):
            if imagine_dropout == 0:
                imagine = False
            elif imagine_dropout == 1:
                imagine = True
            else:
                imagine = np.random.rand() < imagine_dropout
            if not imagine:
                post, (h, z) = self.cell.forward(embeds[i], actions[i], reset_masks[i], (h, z), noises[i])
            else:
                post, (h, z) = self.cell.forward_prior(actions[i], reset_masks[i], (h, z), noises[i])  # post=prior in this case
            posts.append(post)
            states_h.append(h)
            samples.append(z)

        posts = torch.stack(posts)          # (N,BI,2S)
        states_h = torch.stack(states_h)    # (N,BI,D)
        samples = torch.stack(samples)      # (N,BI,S)
        priors = self.cell.batch_prior(states_h)  # (N,BI,2S)
        features = self.to_feature(states_h, samples)   # (N,BI,D+S)

        posts = posts.reshape(n, b, I, -1)  # (N,BI,X) => (N,B,I,X)
        states_h = states_h.reshape(n, b, I, -1)
        samples = samples.reshape(n, b, I, -1)
        priors = priors.reshape(n, b, I, -1)
        states = (states_h, samples)
        features = features.reshape(n, b, I, -1)

        return (
            priors,                      # tensor(N,B,I,2S)
            posts,                       # tensor(N,B,I,2S)
            samples,                     # tensor(N,B,I,S)
            features,                    # tensor(N,B,I,D+S)
            states,
            (h.detach(), z.detach()),
        )

    def init_state(self, batch_size):
        return self.cell.init_state(batch_size)

    def generate_noises(self, n_steps: int, batch_shape: Tuple, device) -> List[Tensor]:
        # TODO: not needed if conf.stoch_discrete
        shape = (n_steps, ) + batch_shape + (self.cell.stoch_dim, )
        noises = torch.normal(torch.zeros(shape), torch.ones(shape)).to(device)
        return noises.unbind(0)  # type: ignore

    def to_feature(self, h: Tensor, z: Tensor) -> Tensor:
        return torch.cat((h, z), -1)

    def feature_replace_z(self, features: Tensor, z: Tensor):
        h, _ = features.split([self.cell.deter_dim, z.shape[-1]], -1)
        return self.to_feature(h, z)

    def zdistr(self, pp: Tensor) -> D.Distribution:
        return self.cell.zdistr(pp)


class RSSMCell(nn.Module):

    def __init__(self, embed_dim, action_dim, deter_dim, stoch_dim, hidden_dim, global_dim, gru_layers, gru_type, layer_norm):
        super().__init__()
        self.stoch_dim = stoch_dim
        self.deter_dim = deter_dim
        self.global_dim = global_dim
        norm = nn.LayerNorm if layer_norm else NoNorm

        self.z_mlp = nn.Linear(stoch_dim, hidden_dim)
        self.a_mlp = nn.Linear(action_dim, hidden_dim, bias=False)  # No bias, because outputs are added
        self.in_norm = norm(hidden_dim, eps=1e-3)
        # self.g_mlp = nn.Linear(global_dim, hidden_dim, bias=False)

        self.gru = my.GRUCellStack(hidden_dim, deter_dim, gru_layers, gru_type)

        self.prior_mlp_h = nn.Linear(deter_dim, hidden_dim)
        self.prior_norm = norm(hidden_dim, eps=1e-3)
        self.prior_mlp = nn.Linear(hidden_dim, 2 * stoch_dim)

        self.post_mlp_h = nn.Linear(deter_dim, hidden_dim)
        self.post_mlp_e = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.post_norm = norm(hidden_dim, eps=1e-3)
        self.post_mlp = nn.Linear(hidden_dim, 2 * stoch_dim)

    def init_state(self, batch_size):
        device = next(self.gru.parameters()).device
        return (
            torch.zeros((batch_size, self.deter_dim), device=device),
            torch.zeros((batch_size, self.stoch_dim), device=device),
        )

    def forward(self,
                embed: Tensor,                    # tensor(B,E)
                action: Tensor,                   # tensor(B,A)
                reset_mask: Tensor,               # tensor(B,1)
                in_state: Tuple[Tensor, Tensor],  # tensor(B,D+S)
                noise: Tensor,                    # tensor(B,S)
                ) -> Tuple[Tensor,
                           Tuple[Tensor, Tensor]]:

        in_h, in_z = in_state
        in_h = in_h * reset_mask
        in_z = in_z * reset_mask

        x = self.z_mlp(in_z) + self.a_mlp(action)  # (B,H)
        x = self.in_norm(x)
        za = F.elu(x)
        h = self.gru(za, in_h)                                             # (B, D)

        x = self.post_mlp_h(h) + self.post_mlp_e(embed)
        x = self.post_norm(x)
        post_in = F.elu(x)
        post = self.post_mlp(post_in)                                    # (B, 2*S)
        sample = rsample(post, noise)                                     # (B, S)

        return (
            post,                         # tensor(B, 2*S)
            (h, sample),                  # tensor(B, D+S+G)
        )

    def forward_prior(self,
                      action: Tensor,                   # tensor(B,A)
                      reset_mask: Optional[Tensor],               # tensor(B,1)
                      in_state: Tuple[Tensor, Tensor],  # tensor(B,D+S)
                      noise: Tensor,                    # tensor(B,S)
                      ) -> Tuple[Tensor,
                                 Tuple[Tensor, Tensor]]:

        in_h, in_z = in_state
        if reset_mask is not None:
            in_h = in_h * reset_mask
            in_z = in_z * reset_mask

        x = self.z_mlp(in_z) + self.a_mlp(action)  # (B,H)
        x = self.in_norm(x)
        za = F.elu(x)
        h = self.gru(za, in_h)                  # (B, D)

        x = self.prior_mlp_h(h)
        x = self.prior_norm(x)
        x = F.elu(x)
        prior = self.prior_mlp(x)          # (B,2S)
        sample = rsample(prior, noise)      # (B,S)

        return (
            prior,                        # (B,2S)
            (h, sample),                  # (B,D+S)
        )

    def batch_prior(self,
                    h: Tensor,     # tensor(N, B, D)
                    ) -> Tensor:
        x = self.prior_mlp_h(h)
        x = self.prior_norm(x)
        x = F.elu(x)
        prior = self.prior_mlp(x)  # tensor(B,2S)
        return prior

    def zdistr(self, pp: Tensor) -> D.Distribution:
        # pp = post or prior
        return diag_normal(pp)


class RSSMCellDiscrete(nn.Module):

    def __init__(self, embed_dim, action_dim, deter_dim, stoch_dim, stoch_discrete, hidden_dim, global_dim, gru_layers, gru_type, layer_norm):
        super().__init__()
        self.stoch_dim = stoch_dim
        self.stoch_discrete = stoch_discrete
        self.deter_dim = deter_dim
        self.global_dim = global_dim
        norm = nn.LayerNorm if layer_norm else NoNorm

        self.z_mlp = nn.Linear(stoch_dim * stoch_discrete, hidden_dim)
        self.a_mlp = nn.Linear(action_dim, hidden_dim, bias=False)  # No bias, because outputs are added
        self.in_norm = norm(hidden_dim, eps=1e-3)
        # self.g_mlp = nn.Linear(global_dim, hidden_dim, bias=False)

        self.gru = my.GRUCellStack(hidden_dim, deter_dim, gru_layers, gru_type)

        self.prior_mlp_h = nn.Linear(deter_dim, hidden_dim)
        self.prior_norm = norm(hidden_dim, eps=1e-3)
        self.prior_mlp = nn.Linear(hidden_dim, stoch_dim * stoch_discrete)

        self.post_mlp_h = nn.Linear(deter_dim, hidden_dim)
        self.post_mlp_e = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.post_norm = norm(hidden_dim, eps=1e-3)
        self.post_mlp = nn.Linear(hidden_dim, stoch_dim * stoch_discrete)

    def init_state(self, batch_size):
        device = next(self.gru.parameters()).device
        return (
            torch.zeros((batch_size, self.deter_dim), device=device),
            torch.zeros((batch_size, self.stoch_dim * self.stoch_discrete), device=device),
        )

    def forward(self,
                embed: Tensor,                    # tensor(B,E)
                action: Tensor,                   # tensor(B,A)
                reset_mask: Tensor,               # tensor(B,1)
                in_state: Tuple[Tensor, Tensor],
                _noise: Tensor,                    # tensor(B,S)
                ) -> Tuple[Tensor,
                           Tuple[Tensor, Tensor]]:

        in_h, in_z = in_state
        in_h = in_h * reset_mask
        in_z = in_z * reset_mask
        B = action.shape[0]

        x = self.z_mlp(in_z) + self.a_mlp(action)  # (B,H)
        x = self.in_norm(x)
        za = F.elu(x)
        h = self.gru(za, in_h)                                             # (B, D)

        x = self.post_mlp_h(h) + self.post_mlp_e(embed)
        x = self.post_norm(x)
        post_in = F.elu(x)
        post = self.post_mlp(post_in)                                    # (B, S*S)
        post_distr = self.zdistr(post)
        sample = post_distr.rsample().reshape(B, -1)

        return (
            post,                         # tensor(B, 2*S)
            (h, sample),                  # tensor(B, D+S+G)
        )

    def forward_prior(self,
                      action: Tensor,                   # tensor(B,A)
                      reset_mask: Optional[Tensor],               # tensor(B,1)
                      in_state: Tuple[Tensor, Tensor],  # tensor(B,D+S)
                      _noise: Tensor,                    # tensor(B,S)
                      ) -> Tuple[Tensor,
                                 Tuple[Tensor, Tensor]]:

        in_h, in_z = in_state
        if reset_mask is not None:
            in_h = in_h * reset_mask
            in_z = in_z * reset_mask

        B = action.shape[0]

        x = self.z_mlp(in_z) + self.a_mlp(action)  # (B,H)
        x = self.in_norm(x)
        za = F.elu(x)
        h = self.gru(za, in_h)                  # (B, D)

        x = self.prior_mlp_h(h)
        x = self.prior_norm(x)
        x = F.elu(x)
        prior = self.prior_mlp(x)          # (B,2S)
        prior_distr = self.zdistr(prior)
        sample = prior_distr.rsample().reshape(B, -1)

        return (
            prior,                        # (B,2S)
            (h, sample),                  # (B,D+S)
        )

    def batch_prior(self,
                    h: Tensor,     # tensor(N, B, D)
                    ) -> Tensor:
        x = self.prior_mlp_h(h)
        x = self.prior_norm(x)
        x = F.elu(x)
        prior = self.prior_mlp(x)  # tensor(B,2S)
        return prior

    def zdistr(self, pp: Tensor) -> D.Distribution:
        # pp = post or prior
        logits = pp.reshape(pp.shape[:-1] + (self.stoch_dim, self.stoch_discrete))
        distr = D.OneHotCategoricalStraightThrough(logits=logits.float())  # NOTE: .float() needed to force float32 on AMP
        distr = D.independent.Independent(distr, 1)  # This makes d.entropy() and d.kl() sum over stoch_dim
        return distr
