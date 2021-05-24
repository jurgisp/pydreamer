import torch
import torch.nn as nn
import torch.distributions as D

from modules_tools import *


class ConvEncoder(nn.Module):

    def __init__(self, in_channels=3, kernels=(4, 4, 4, 4), stride=2, out_dim=256, activation=nn.ELU):
        super().__init__()
        self.out_dim = out_dim
        assert out_dim == 256
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
        return self._model(x)


class ConvDecoderCat(nn.Module):

    def __init__(self, in_dim, out_channels=3, kernels=(5, 5, 6, 6), stride=2, activation=nn.ELU):
        super().__init__()
        self.in_dim = in_dim
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
        return self._model(x)

    def loss(self, output, target):
        n = output.size(0)
        output = flatten(output)
        target = flatten(target).argmax(dim=-3)
        loss = F.cross_entropy(output, target, reduction='none')
        loss = unflatten(loss, n)
        return loss.sum(dim=[-1, -2])


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
        return self._model(x)


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

    def forward(self, x):
        assert len(x.shape) == 2
        return self._model(x)

    def loss(self,
             output,  # (NB,C,H,W)
             target   # float(NB,C,H,W) or int(NB,H,W)
             ):
        if output.shape == target.shape:
            target = target.argmax(dim=-3)  # float(NB,C,H,W) => int(NB,H,W)

        if self._min_prob == 0:
            loss = F.nll_loss(F.log_softmax(output, 1), target, reduction='none')  # = F.cross_entropy()
        else:
            prob = F.softmax(output, 1)
            prob = (1.0 - self._min_prob) * prob + self._min_prob * (1.0 / prob.size(1))  # mix with uniform prob
            loss = F.nll_loss(prob.log(), target, reduction='none')
        
        return loss.sum(dim=[-1, -2])  # (NB,H,W) => (NB)


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

    def predict_obs(self,
                    prior, post, obs_rec, states,                 # forward() output
                    ):
        n = prior.size(0)
        # Sample from prior instead of posterior
        sample = diag_normal(prior).sample()
        obs_pred = unflatten(self._decoder(flatten(cat(states, sample))), n)    # (N,B,C,MH,MW)
        obs_pred_distr = D.Categorical(logits=obs_pred.permute(0, 1, 3, 4, 2))  # (N,B,C,MH,MW) => (N,B,MH,MW,C)
        return obs_pred_distr       # categorical(N,B,HM,WM,C)


class DirectHead(nn.Module):

    def __init__(self, decoder: DenseDecoder):
        super().__init__()
        self._decoder = decoder

    def forward(self,
                # obs,       # tensor(B, C, H, W)
                state,     # tensor(B, D+S+G)
                ):
        assert len(state.shape) == 2
        return self._decoder(state)  # TODO: make VAE head -compatible

    def loss(self,
             obs_pred,          # forward() output
             obs_target,        # tensor(N, B, C, H, W)
             ):
        return self._decoder.loss(obs_pred, obs_target)  # TODO: make VAE head -compatible

    def predict_obs(self,
                    obs_pred,                 # forward() output
                    ):
        return imgrec_to_distr(obs_pred)


class NoHead(nn.Module):

    def __init__(self, out_shape):
        super().__init__()
        self.out_shape = out_shape  # (C,MH,MW)

    def forward(self, obs, state):
        return (obs,)

    def loss(self, obs, obs_target):
        return torch.tensor(0.0, device=obs.device), {}

    def predict_obs(self, obs):
        zeros = torch.zeros(obs.shape[:2] + self.out_shape, device=obs.device)  # (N,B,C,MH,MW)
        return D.Categorical(logits=zeros.permute(0, 1, 3, 4, 2))  # (N,B,MH,MW,C)
