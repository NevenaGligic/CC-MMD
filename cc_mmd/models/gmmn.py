"""Convolutional generator and MLP discriminator used in Experiment 4 (GMMN and GAN).

The DCGAN-style generator follows the ConvGenerator of the GMMN / MMD-GAN
literature (github.com/OctoberChang/MMD-GAN), with outputs in logit space.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

DIM: int = 784   # flattened 28x28 image


class ConvGenerator(nn.Module):
    """z -> FC -> (256, 7, 7) -> two transposed convolutions -> 3x3 convolution -> 784 logits.

    Args:
        z_dim: Dimensionality of the input noise.
    """

    def __init__(self, z_dim: int = 128) -> None:
        super().__init__()
        self.z_dim = z_dim

        self.fc = nn.Sequential(
            nn.Linear(z_dim, 256 * 7 * 7),
            nn.BatchNorm1d(256 * 7 * 7),
            nn.ReLU(True),
        )
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, z: Tensor) -> Tensor:
        """Map noise of shape (B, z_dim) to logit-space outputs of shape (B, 784)."""
        x = self.fc(z).view(-1, 256, 7, 7)
        return self.conv(x).view(-1, DIM)


class Discriminator(nn.Module):
    """MLP discriminator (dim -> hidden_dim -> hidden_dim/2 -> 1, LeakyReLU(0.2)) returning logits.

    Args:
        dim: Input dimensionality.
        hidden_dim: Width of the first hidden layer.
    """

    def __init__(self, dim: int = DIM, hidden_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def make_discriminator(dim: int = DIM, hidden_dim: int = 512) -> Discriminator:
    """Build a Discriminator (on CPU)."""
    return Discriminator(dim=dim, hidden_dim=hidden_dim)


def make_gen(z_dim: int = 128) -> ConvGenerator:
    """Build a ConvGenerator (on CPU)."""
    return ConvGenerator(z_dim=z_dim)


def gen_sample(gen: nn.Module, n: int, device: torch.device) -> Tensor:
    """Generate n samples from z ~ N(0, I), keeping the graph for gradient-based training."""
    z = torch.randn(n, gen.z_dim, device=device)
    return gen(z)
