"""Image preprocessing shared by MNIST, K-MNIST and Omniglot (Experiment 4).

Every 28x28 grayscale image is mapped to a 784-dimensional logit-space vector:
  ToTensor -> clamp to [ALPHA, 1 - ALPHA] -> logit -> flatten
"""

from __future__ import annotations

import torch
import torchvision.transforms as T
from torch import Tensor

# Keeps the logit finite at pixel values 0 and 1.
ALPHA: float = 0.05


class LogitTransform:
    """Clamp a [0, 1] tensor to [ALPHA, 1 - ALPHA] and apply the logit element-wise."""

    def __call__(self, x: Tensor) -> Tensor:
        x = x.clamp(ALPHA, 1.0 - ALPHA)
        return torch.log(x / (1.0 - x))


def get_mnist_transform() -> T.Compose:
    """Transform producing a (784,) logit-space tensor from a 28x28 grayscale image."""
    return T.Compose([
        T.ToTensor(),
        LogitTransform(),
        T.Lambda(lambda x: x.view(-1)),
    ])
