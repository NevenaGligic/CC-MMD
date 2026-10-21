"""Variational autoencoder (VAE) used in Experiment 4.

Encoder: in_dim -> hidden_dim -> hidden_dim/2 -> hidden_dim/4 -> (mu, log_var) in R^latent_dim
Decoder: latent_dim -> hidden_dim/4 -> hidden_dim/2 -> hidden_dim -> in_dim (no final activation)
All hidden layers use ReLU; inputs and outputs are in logit space.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

DIM: int = 784   # flattened 28x28 image


class _Encoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        mid = hidden_dim // 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, mid),
            nn.ReLU(),
            nn.Linear(mid, mid // 2),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(mid // 2, latent_dim)
        self.fc_logvar = nn.Linear(mid // 2, latent_dim)

    def forward(self, x: Tensor):
        h = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)


class _Decoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        mid = hidden_dim // 2
        self.net = nn.Sequential(
            nn.Linear(latent_dim, mid // 2),
            nn.ReLU(),
            nn.Linear(mid // 2, mid),
            nn.ReLU(),
            nn.Linear(mid, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)


class VAE(nn.Module):
    """VAE with a diagonal-Gaussian approximate posterior.

    Args:
        in_dim: Input and output dimensionality.
        hidden_dim: Width of the first encoder layer (and the last decoder hidden layer).
        latent_dim: Dimensionality of the latent code z.
    """

    def __init__(self, in_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = _Encoder(in_dim, hidden_dim, latent_dim)
        self.decoder = _Decoder(latent_dim, hidden_dim, in_dim)

    def encode(self, x: Tensor):
        """Return (mu, log_var) of q(z | x)."""
        return self.encoder(x)

    def reparametrize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Sample z ~ q(z | x) in training mode; return the mean in eval mode."""
        if self.training:
            std = (0.5 * logvar).exp()
            return mu + std * torch.randn_like(std)
        return mu

    def decode(self, z: Tensor) -> Tensor:
        """Map latent codes to logit-space vectors."""
        return self.decoder(z)

    def forward(self, x: Tensor):
        """Return (reconstruction, mu, log_var)."""
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar


def make_vae(in_dim: int, hidden_dim: int, latent_dim: int) -> VAE:
    """Build a VAE (on CPU)."""
    return VAE(in_dim=in_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)


def vae_sample(vae: nn.Module, n: int, device: torch.device) -> Tensor:
    """Decode n draws from the prior N(0, I), keeping the graph for gradient-based training."""
    z = torch.randn(n, vae.latent_dim, device=device)
    return vae.decode(z)
