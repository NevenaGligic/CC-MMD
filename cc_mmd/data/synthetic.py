"""Synthetic data for Experiment 1 (MMD estimation) and Experiment 2 (linear regression)."""

from typing import Tuple

import numpy as np
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Experiment 1: ring-of-Gaussians + gradient contamination
# ---------------------------------------------------------------------------

def sample_ring_gaussians(
    n: int,
    *,
    R: float = 5.0,
    sigma_q: float = 0.6,
    K: int = 5,
    rng: torch.Generator,
    device: torch.device,
) -> Tensor:
    """Sample from a mixture of K Gaussian clusters on a circle.

    Cluster centers are equally spaced on a circle of radius R. Each
    component has isotropic covariance sigma_q^2 * I_2 and equal weight.

    Args:
        n: Number of samples.
        R: Circle radius.
        sigma_q: Standard deviation of each Gaussian cluster.
        K: Number of clusters.
        rng: PyTorch random generator for reproducibility.
        device: Target device.

    Returns:
        Tensor of shape (n, 2).
    """
    angles = torch.linspace(0.0, 2 * torch.pi * (1 - 1 / K), K, device=device)
    centers = torch.stack([R * angles.cos(), R * angles.sin()], dim=1)  # (K, 2)

    # Assign each sample to a cluster uniformly at random
    cluster_ids = torch.randint(0, K, (n,), generator=rng, device=device)
    noise = torch.randn(n, 2, generator=rng, device=device) * sigma_q
    return centers[cluster_ids] + noise


def sample_nuisance_gradient(
    n: int,
    *,
    rng: torch.Generator,
    device: torch.device,
) -> Tensor:
    """Sample from the gradient nuisance distribution via rejection sampling.

    Candidate points are drawn from Uniform([-8, 8]^2). Each is accepted
    with probability alpha(x) = 0.25 + 0.75 * (x + 8) / 16, where x is
    the first coordinate. This induces a left-to-right gradient.

    Args:
        n: Number of accepted samples required.
        rng: PyTorch random generator.
        device: Target device.

    Returns:
        Tensor of shape (n, 2).
    """
    accepted = []
    total = 0
    while total < n:
        batch = 4 * n  # oversample to amortise rejection overhead
        candidates = torch.rand(batch, 2, generator=rng, device=device) * 16.0 - 8.0
        x = candidates[:, 0]
        alpha = 0.25 + 0.75 * (x + 8.0) / 16.0
        u = torch.rand(batch, generator=rng, device=device)
        mask = u < alpha
        accepted.append(candidates[mask])
        total += int(mask.sum().item())

    return torch.cat(accepted, dim=0)[:n]


def make_contaminated_pool(
    N: int,
    epsilon: float,
    *,
    R: float = 5.0,
    sigma_q: float = 0.6,
    K: int = 5,
    rng: torch.Generator,
    device: torch.device,
) -> Tuple[Tensor, Tensor]:
    """Generate the contaminated pooled sample.

    Each point is drawn from q with probability (1 - epsilon) and from z
    with probability epsilon. The latent membership indicator I_j = 1 iff
    the point comes from q.

    Args:
        N: Total pool size.
        epsilon: Contamination rate in [0, 1).
        R: Circle radius for q.
        sigma_q: Cluster std for q.
        K: Number of clusters in q.
        rng: PyTorch random generator.
        device: Target device.

    Returns:
        Tuple (W, I):
            W — pooled sample, shape (N, 2).
            I — binary membership indicators, shape (N,), dtype torch.int32.
    """
    # Determine membership
    u = torch.rand(N, generator=rng, device=device)
    I = (u >= epsilon).to(torch.int32)

    n_clean = int(I.sum().item())
    n_contam = N - n_clean

    W_clean = sample_ring_gaussians(n_clean, R=R, sigma_q=sigma_q, K=K, rng=rng, device=device)

    W = torch.empty(N, 2, device=device)
    W[I.bool()] = W_clean
    if n_contam > 0:
        W_contam = sample_nuisance_gradient(n_contam, rng=rng, device=device)
        W[~I.bool()] = W_contam
    return W, I


def compute_proxy_scores_exp1(
    W: Tensor,
    *,
    R: float = 5.0,
    sigma_cls: float = 1.0,
    rng: torch.Generator,
) -> Tensor:
    """Compute weak proxy membership scores for Experiment 1.

    Uses radial distance and horizontal coordinate as features, plus
    Gaussian noise, passed through a logistic function.

    s_j = 1.5 - 0.9 * |‖W_j‖ - R| - 0.20 * W_j[0] + N(0, sigma_cls^2)
    I_tilde_j = sigmoid(s_j)

    Args:
        W: Pooled sample, shape (N, 2).
        R: Circle radius.
        sigma_cls: Standard deviation of the classifier noise (default 1.0).
            Larger values weaken the proxy; 0.0 gives a noiseless proxy.
        rng: PyTorch random generator.

    Returns:
        Proxy scores in [0, 1], shape (N,).
    """
    r = W.norm(dim=1)
    x = W[:, 0]
    noise = torch.randn(W.shape[0], generator=rng, device=W.device) * sigma_cls
    s = 1.5 - 0.9 * (r - R).abs() - 0.20 * x + noise
    return torch.sigmoid(s)


# ---------------------------------------------------------------------------
# Experiment 2: contaminated linear regression
# ---------------------------------------------------------------------------

def generate_regression_data(
    N: int,
    epsilon: float,
    beta: Tuple[float, float],
    sigma: float,
    *,
    K_clusters: int = 4,
    rng: torch.Generator,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Generate contaminated regression data.

    Inlier model: Y = beta[0] + beta[1]*X + N(0, sigma^2),
    X ~ Uniform[-3, 3].

    Contamination: K_clusters Gaussian clusters in (x, y) space with
    centers drawn from Uniform[-4.5, 4.5] x Uniform[-8, 8] and
    sigma_x=0.35, sigma_y=0.75.

    Args:
        N: Total dataset size.
        epsilon: Contamination rate in [0, 1).
        beta: True regression parameters (intercept, slope).
        sigma: Inlier noise standard deviation.
        K_clusters: Number of contamination clusters.
        rng: PyTorch random generator.
        device: Target device.

    Returns:
        Tuple (X_data, Y_data, I, cluster_centers):
            X_data — covariates, shape (N,).
            Y_data — responses, shape (N,).
            I — binary membership indicators, shape (N,), 1 = inlier.
            cluster_centers — contamination cluster centers, shape (K, 2).
    """
    beta0, beta1 = beta

    # Membership
    u = torch.rand(N, generator=rng, device=device)
    I = (u >= epsilon).to(torch.int32)
    n_clean = int(I.sum().item())
    n_contam = N - n_clean

    # Inlier data
    X_clean = torch.rand(n_clean, generator=rng, device=device) * 6.0 - 3.0
    noise = torch.randn(n_clean, generator=rng, device=device) * sigma
    Y_clean = beta0 + beta1 * X_clean + noise

    # Contamination cluster centers
    cx = torch.rand(K_clusters, generator=rng, device=device) * 9.0 - 4.5
    cy = torch.rand(K_clusters, generator=rng, device=device) * 16.0 - 8.0
    cluster_centers = torch.stack([cx, cy], dim=1)  # (K, 2)

    # Contaminant data
    cluster_ids = torch.randint(0, K_clusters, (n_contam,), generator=rng, device=device)
    X_contam = cx[cluster_ids] + torch.randn(n_contam, generator=rng, device=device) * 0.35
    Y_contam = cy[cluster_ids] + torch.randn(n_contam, generator=rng, device=device) * 0.75

    # Assemble
    X_data = torch.empty(N, device=device)
    Y_data = torch.empty(N, device=device)
    X_data[I.bool()] = X_clean
    X_data[~I.bool()] = X_contam
    Y_data[I.bool()] = Y_clean
    Y_data[~I.bool()] = Y_contam

    return X_data, Y_data, I, cluster_centers


def compute_proxy_scores_regression(
    X_data: Tensor,
    Y_data: Tensor,
    *,
    rng: torch.Generator,
) -> Tensor:
    """Compute weak proxy membership scores for the regression experiment.

    Fits OLS on the full (contaminated) dataset, computes residuals,
    robustly standardises by MAD, then applies:
        I_tilde_j = sigmoid(1.5 - 1.2 * z_j + N(0, 0.9^2))

    Args:
        X_data: Covariates, shape (N,).
        Y_data: Responses, shape (N,).
        rng: PyTorch random generator.

    Returns:
        Proxy scores in [0, 1], shape (N,).
    """
    # OLS on contaminated data
    X_np = X_data.cpu().numpy()
    Y_np = Y_data.cpu().numpy()
    A = np.column_stack([np.ones_like(X_np), X_np])
    beta_ols, _, _, _ = np.linalg.lstsq(A, Y_np, rcond=None)
    residuals = Y_np - (beta_ols[0] + beta_ols[1] * X_np)

    # MAD-based robust scale
    med = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - med)))
    sigma_mad = 1.4826 * mad if mad > 0 else 1.0
    z = np.abs(residuals) / sigma_mad

    z_tensor = torch.from_numpy(z).to(dtype=X_data.dtype, device=X_data.device)
    noise = torch.randn(X_data.shape[0], generator=rng, device=X_data.device) * 0.9
    s = 1.5 - 1.2 * z_tensor + noise
    return torch.sigmoid(s)
