"""Sliced Wasserstein distance (SWD) training loss, the IAF baseline in Experiment 4."""

from __future__ import annotations

import torch
from torch import Tensor


class SWDLoss:
    """Squared sliced Wasserstein distance between generated samples and the pool W.

    Each call subsamples W to the generated batch size, projects both sets onto
    n_projections random unit vectors, and returns the mean squared difference
    between the sorted projections.

    Args:
        W: Training pool, shape (N, d).
        n_projections: Number of random one-dimensional projections.
    """

    def __init__(self, W: Tensor, n_projections: int) -> None:
        self.W = W
        self.n_projections = n_projections

    def __call__(self, X_gen: Tensor) -> Tensor:
        """SWD^2 for generated samples X_gen of shape (n, d); differentiable in X_gen."""
        n, d = X_gen.shape

        idx = torch.randperm(self.W.shape[0], device=self.W.device)[:n]
        W_sub = self.W[idx]

        projs = torch.randn(d, self.n_projections, device=X_gen.device)
        projs = projs / projs.norm(dim=0, keepdim=True)

        X_sorted = (X_gen @ projs).sort(dim=0).values
        W_sorted = (W_sub @ projs).sort(dim=0).values.detach()

        return ((X_sorted - W_sorted) ** 2).mean()
