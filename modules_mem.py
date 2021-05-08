import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from modules_tools import *


class NoMemory(nn.Module):

    def __init__(self):
        super().__init__()
        self.global_dim = 0
        self.register_buffer('_empty', torch.FloatTensor(), persistent=False)  # Gets moved to GPU automatically
        self.register_buffer('_zero', torch.tensor(0.0), persistent=False)

    def forward(self, embed, action, reset, in_state):
        return (in_state,)

    def init_state(self, batch_size):
        return self._empty

    def loss(self, *args):
        return self._zero


class GlobalStateMem(nn.Module):

    def __init__(self, embed_dim=256, action_dim=7, mem_dim=200, stoch_dim=30, hidden_dim=200, loss_type='last_kl'):
        super().__init__()
        self._cell = GlobalStateCell(embed_dim, action_dim, mem_dim, stoch_dim, hidden_dim)
        self.global_dim = stoch_dim
        self.loss_type = loss_type

    def forward(self,
                embed,     # tensor(N, B, E)
                action,    # tensor(N, B, A)
                reset,     # tensor(N, B)
                in_state_post,  # (tensor(B, M), tensor(B, S))
                ):

        n = embed.size(0)
        states = []
        posts = []
        state = in_state_post[0]

        for i in range(n):
            state, post = self._cell(embed[i], action[i], reset[i], state)
            states.append(state)
            posts.append(post)

        sample = diag_normal(posts[-1]).rsample()
        out_state_post = (states[-1].detach(), posts[-1].detach())

        return (
            sample,                      # tensor(   B, S)
            torch.stack(states),         # tensor(N, B, M)
            torch.stack(posts),          # tensor(N, B, 2S)
            in_state_post,               # (tensor(B, M), tensor(B, 2S)) - for loss
            out_state_post,              # (tensor(B, M), tensor(B, 2S))
        )

    def init_state(self, batch_size):
        return self._cell.init_state(batch_size)

    def loss(self,
             sample, states, posts, in_state_post, out_state_post,       # forward() output
             ):
        if self.loss_type == 'chained_kl':
            in_post = in_state_post[1]
            priors = torch.cat([in_post.unsqueeze(0), posts[:-1]])
            loss_kl = D.kl.kl_divergence(diag_normal(posts), diag_normal(priors))  # KL between consecutive posteriors
            loss_kl = loss_kl.mean()        # (N, B) => ()
            return loss_kl

        if self.loss_type == 'last_kl':
            post = posts[-1]  # (B, 2S)
            prior = torch.zeros_like(post)
            loss_kl = D.kl.kl_divergence(diag_normal(post), diag_normal(prior))
            # Divide by N, because loss_kl is for the whole sequence, and returned loss is assumed per-step
            loss_kl = loss_kl.mean() / posts.shape[0]  # (B) => ()
            return loss_kl

        assert False



class GlobalStateCell(nn.Module):

    def __init__(self, embed_dim=256, action_dim=7, mem_dim=200, stoch_dim=30, hidden_dim=200):
        super().__init__()
        self._mem_dim = mem_dim

        self._ea_mlp = nn.Sequential(nn.Linear(embed_dim + action_dim, hidden_dim),
                                     nn.ELU())

        self._gru = nn.GRUCell(hidden_dim, mem_dim)

        self._post_mlp = nn.Sequential(nn.Linear(mem_dim, hidden_dim),
                                       nn.ELU(),
                                       nn.Linear(hidden_dim, 2 * stoch_dim))

    def init_state(self, batch_size):
        device = next(self._gru.parameters()).device
        state = torch.zeros((batch_size, self._mem_dim), device=device)
        post = self._post_mlp(state)
        return (state.detach(), post.detach())

    def forward(self,
                embed,     # tensor(B, E)
                action,    # tensor(B, A)
                reset,     # tensor(B)
                in_state,  # tensor(B, M)
                ):

        in_state = in_state * ~reset.unsqueeze(1)

        ea = self._ea_mlp(cat(embed, action))                                # (B, H)
        state = self._gru(ea, in_state)                                     # (B, M)
        post = self._post_mlp(state)                                    # (B, 2*S)

        return (
            state,           # tensor(B, M)
            post,            # tensor(B, 2*S)
        )
