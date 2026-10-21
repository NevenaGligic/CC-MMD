"""Kernel functions for MMD computation."""

import torch
from torch import Tensor


def gaussian_kernel(X: Tensor, Y: Tensor, sigma: float) -> Tensor:
    """Compute the Gaussian (RBF) kernel matrix between two sets of points.

    Args:
        X: Tensor of shape (n, d).
        Y: Tensor of shape (m, d).
        sigma: Bandwidth parameter. The kernel is
            k(x, y) = exp(-||x - y||^2 / (2 * sigma^2)).

    Returns:
        Kernel matrix of shape (n, m) on the same device as X.
    """
    sq_dists = torch.cdist(X, Y, p=2).pow(2)
    return torch.exp(-sq_dists / (2.0 * sigma ** 2))
