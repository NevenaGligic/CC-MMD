"""Generative models for Experiment 4: IAF (flow.py), VAE (vae.py) and a convolutional generator (gmmn.py).

All models produce 784-dimensional vectors in logit space.
"""

import torch
from torch import Tensor


def to_images(x: Tensor) -> Tensor:
    """Map logit-space outputs of shape (B, 784) to images of shape (B, 1, 28, 28) in [0, 1]."""
    return torch.sigmoid(x).reshape(x.shape[0], 1, 28, 28)
