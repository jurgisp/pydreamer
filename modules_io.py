import torch
import torch.nn as nn
import torch.distributions as D

from modules_tools import *


class ConvEncoder(nn.Module):

    def __init__(self, in_channels=3, out_dim=1024, activation=nn.ELU):
        super().__init__()
        self.out_dim = out_dim
        assert out_dim == 1024
        kernels = (4, 4, 4, 4)
        stride = 2
        self._model = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernels[0], stride),
            activation(),
            nn.Conv2d(32, 64, kernels[1], stride),
            activation(),
            nn.Conv2d(64, 128, kernels[2], stride),
            activation(),
            nn.Conv2d(128, 256, kernels[3], stride),
            activation(),
            nn.Flatten()
        )

    def forward(self, x):
        x, bd = flatten_batch(x, 3)
        y = self._model(x)
        y = unflatten_batch(y, bd)
        return y


class ConvDecoder(nn.Module):

    def __init__(self, in_dim, out_channels=3, activation=nn.ELU):
        super().__init__()
        self.in_dim = in_dim
        kernels = (5, 5, 6, 6)
        stride = 2
        self._model = nn.Sequential(
            # FC
            nn.Linear(in_dim, 1024),  # No activation here in DreamerV2
            nn.Unflatten(-1, (1024, 1, 1)),  # type: ignore
            # Deconv
            nn.ConvTranspose2d(1024, 128, kernels[0], stride),
            activation(),
            nn.ConvTranspose2d(128, 64, kernels[1], stride),
            activation(),
            nn.ConvTranspose2d(64, 32, kernels[2], stride),
            activation(),
            nn.ConvTranspose2d(32, out_channels, kernels[3], stride))

    def forward(self, x):
        x, bd = flatten_batch(x)
        y = self._model(x)
        y = unflatten_batch(y, bd)
        return y

    def loss(self, output, target):
        output, bd = flatten_batch(output, 3)
        target, _ = flatten_batch(target, 3)
        loss = torch.square(output - target).sum(dim=[-1, -2, -3])  # MSE
        return unflatten_batch(loss, bd)

    def to_distr(self, output: Tensor) -> D.Categorical:
        # TODO
        assert len(output.shape) == 6
        logits = output.permute(2, 0, 1, 4, 5, 3)  # (N,B,I,C,H,W) => (I,N,B,H,W,C)
        # Normalize probability
        logits = logits - logits.logsumexp(dim=-1, keepdim=True)
        # Aggregate prob=avg(prob_i)
        logits_agg = torch.logsumexp(logits, dim=0)  # (I,N,B,H,W,C) => (N,B,H,W,C)
        return D.Categorical(logits=logits_agg)


class DenseEncoder(nn.Module):

    def __init__(self, in_dim, out_dim=256, activation=nn.ELU, hidden_dim=400, hidden_layers=2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        layers = [nn.Flatten()]
        layers += [
            nn.Linear(in_dim, hidden_dim),
            activation()]
        for _ in range(hidden_layers - 1):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                activation()]
        layers += [
            nn.Linear(hidden_dim, out_dim),
            activation()]
        self._model = nn.Sequential(*layers)

    def forward(self, x):
        x, bd = flatten_batch(x, 3)
        y = self._model(x)
        y = unflatten_batch(y, bd)
        return y


class DenseDecoder(nn.Module):

    def __init__(self, in_dim, out_shape=(33, 7, 7), activation=nn.ELU, hidden_dim=400, hidden_layers=2, min_prob=0):
        super().__init__()
        self.in_dim = in_dim
        layers = []
        layers += [
            nn.Linear(in_dim, hidden_dim),
            activation()]
        for _ in range(hidden_layers - 1):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                activation()]
        layers += [
            nn.Linear(hidden_dim, np.prod(out_shape)),
            nn.Unflatten(-1, out_shape)]
        self._model = nn.Sequential(*layers)
        self._min_prob = min_prob

    def forward(self, x: Tensor) -> Tensor:
        x, bd = flatten_batch(x)
        y = self._model(x)
        y = unflatten_batch(y, bd)
        return y

    def loss(self,
             output,  # (NB,C,H,W)
             target   # float(NB,C,H,W) or int(NB,H,W)
             ):
        if output.shape == target.shape:
            target = target.argmax(dim=-3)  # float(NB,C,H,W) => int(NB,H,W)
        output, bd = flatten_batch(output, 3)
        target, _ = flatten_batch(target, 2)

        if self._min_prob == 0:
            loss = F.nll_loss(F.log_softmax(output, 1), target, reduction='none')  # = F.cross_entropy()
        else:
            prob = F.softmax(output, 1)
            prob = (1.0 - self._min_prob) * prob + self._min_prob * (1.0 / prob.size(1))  # mix with uniform prob
            loss = F.nll_loss(prob.log(), target, reduction='none')

        loss = loss.sum(dim=[-1, -2])  # (NB,H,W) => (NB)
        return unflatten_batch(loss, bd)

    def to_distr(self, output: Tensor) -> D.Categorical:
        assert len(output.shape) == 6
        logits = output.permute(2, 0, 1, 4, 5, 3)  # (N,B,I,C,H,W) => (I,N,B,H,W,C)
        # Normalize probability
        logits = logits - logits.logsumexp(dim=-1, keepdim=True)
        # Aggregate prob=avg(prob_i)
        logits_agg = torch.logsumexp(logits, dim=0)  # (I,N,B,H,W,C) => (N,B,H,W,C)
        return D.Categorical(logits=logits_agg)



class CondVAEHead(nn.Module):
    # Conditioned VAE

    def __init__(self, encoder, decoder, state_dim=230, hidden_dim=200, latent_dim=30):
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

    def forward(self,
                obs,       # tensor(N, B, C, H, W)
                state,     # tensor(N, B, D+S+G)
                ):
        states_in = state

        n = obs.size(0)
        embed = self._encoder(flatten(obs))
        state = flatten(state)

        prior = self._prior_mlp(state)              # (N*B, 2*Z)
        post = self._post_mlp(cat(state, embed))    # (N*B, 2*Z)
        sample = diag_normal(post).rsample()                                    # (N*B, Z)
        obs_rec = self._decoder(cat(state, sample))

        return (
            unflatten(prior, n),         # tensor(N, B, 2*Z)
            unflatten(post, n),          # tensor(N, B, 2*Z)
            unflatten(obs_rec, n),       # tensor(N, B, C, H, W)
            states_in
        )

    def loss(self,
             prior, post, obs_rec, states,       # forward() output
             obs_target,                         # tensor(N, B, C, H, W)
             ):
        loss_kl = D.kl.kl_divergence(diag_normal(post), diag_normal(prior))
        loss_rec = self._decoder.loss(obs_rec, obs_target)
        assert loss_kl.shape == loss_rec.shape
        loss_kl, loss_rec = loss_kl.mean(), loss_rec.mean()  # (N, B) => ()
        loss = loss_kl + loss_rec
        metrics = dict(loss_kl=loss_kl.detach(), loss_rec=loss_rec.detach())
        return loss, metrics


class DirectHead(nn.Module):

    def __init__(self, decoder: DenseDecoder):
        super().__init__()
        self._decoder = decoder

    def forward(self,
                # obs,       # tensor(B, C, H, W)
                state,     # tensor(B, D+S+G)
                ):
        return self._decoder(state)  # TODO: make VAE head -compatible

    def loss(self,
             obs_pred,          # forward() output
             obs_target,        # tensor(N, B, C, H, W)
             ):
        return self._decoder.loss(obs_pred, obs_target)  # TODO: make VAE head -compatible


class NoHead(nn.Module):

    def __init__(self, out_shape):
        super().__init__()
        self.out_shape = out_shape  # (C,MH,MW)

    def forward(self, obs, state):
        return (obs,)

    def loss(self, obs, obs_target):
        return torch.tensor(0.0, device=obs.device), {}

