"""Evaluation of trained generators against the clean MNIST test set (Experiment 4).

Test MMD^2 is computed in logit space with a Gaussian kernel; FID is computed in
the feature space of the MNIST CNN in fid.py.
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import torch
import torch.nn as nn

from cc_mmd.estimator import naive_mmd2
from cc_mmd.evaluation.fid import MNISTFeatureNet, compute_fid, extract_features
from cc_mmd.kernels import gaussian_kernel
from cc_mmd.models import to_images
from cc_mmd.models.flow import flow_sample
from cc_mmd.models.vae import vae_sample


@torch.no_grad()
def evaluate_vae(
    vae: nn.Module,
    W_test: torch.Tensor,
    sigma_test: float,
    fid_net: MNISTFeatureNet,
    feat_test: np.ndarray,
    n_gen_eval: int,
    device: torch.device,
    n_eval_mmd: int | None = None,
) -> Dict[str, float]:
    """Test MMD^2 and FID of a trained VAE (samples decoded from z ~ N(0, I)).

    Args:
        vae: Trained VAE in eval mode.
        W_test: Clean test images in logit space, shape (n_test, 784).
        sigma_test: Gaussian kernel bandwidth for the test MMD^2.
        fid_net: Trained FID feature extractor.
        feat_test: Precomputed test-set features, shape (n_test, 128).
        n_gen_eval: Number of generated samples for FID.
        device: Target device.
        n_eval_mmd: Number of generated samples for test MMD^2 (defaults to n_gen_eval).

    Returns:
        Dict with keys "test_mmd2" and "fid".
    """
    if n_eval_mmd is None:
        n_eval_mmd = n_gen_eval

    X_mmd = vae_sample(vae, n_eval_mmd, device)
    kernel_fn = lambda A, B: gaussian_kernel(A, B, sigma_test)
    test_mmd2 = naive_mmd2(X_mmd, W_test, kernel_fn).item()

    X_fid = vae_sample(vae, n_gen_eval, device)
    fid = compute_fid(feat_test, extract_features(fid_net, to_images(X_fid)))

    return {"test_mmd2": test_mmd2, "fid": fid}


@torch.no_grad()
def evaluate_flow(
    flow: nn.Module,
    W_test: torch.Tensor,
    sigma_test: float,
    fid_net: MNISTFeatureNet,
    feat_test: np.ndarray,
    n_gen_eval: int,
    device: torch.device,
    n_eval_mmd: int | None = None,
) -> Dict[str, float]:
    """Test MMD^2 and FID of a trained IAF. Arguments as in evaluate_vae."""
    if n_eval_mmd is None:
        n_eval_mmd = n_gen_eval

    X_mmd = flow_sample(flow, n_eval_mmd, device)
    kernel_fn = lambda A, B: gaussian_kernel(A, B, sigma_test)
    test_mmd2 = naive_mmd2(X_mmd, W_test, kernel_fn).item()

    X_fid = flow_sample(flow, n_gen_eval, device)
    fid = compute_fid(feat_test, extract_features(fid_net, to_images(X_fid)))

    return {"test_mmd2": test_mmd2, "fid": fid}


@torch.no_grad()
def evaluate_gen(
    model: nn.Module,
    sample_fn: Callable,
    W_test: torch.Tensor,
    sigma_test: float,
    fid_net: MNISTFeatureNet,
    feat_test: np.ndarray,
    n_gen_eval: int,
    device: torch.device,
    n_eval_mmd: int | None = None,
    sample_batch_size: int = 2000,
) -> Dict[str, float]:
    """Test MMD^2 and FID of a generator sampled with sample_fn, in batches of sample_batch_size.

    Args:
        model: Trained generator in eval mode.
        sample_fn: Callable (model, n, device) -> samples of shape (n, 784) in logit space.
        sample_batch_size: Number of samples generated per forward pass.
        Other arguments as in evaluate_vae.

    Returns:
        Dict with keys "test_mmd2" and "fid".
    """
    if n_eval_mmd is None:
        n_eval_mmd = n_gen_eval

    def _sample_batched(n_total: int) -> torch.Tensor:
        chunks = []
        for i in range(0, n_total, sample_batch_size):
            chunks.append(sample_fn(model, min(sample_batch_size, n_total - i), device))
        return torch.cat(chunks, dim=0)

    X_mmd = _sample_batched(n_eval_mmd)
    kernel_fn = lambda A, B: gaussian_kernel(A, B, sigma_test)
    test_mmd2 = naive_mmd2(X_mmd, W_test, kernel_fn).item()

    X_fid = _sample_batched(n_gen_eval)
    fid = compute_fid(feat_test, extract_features(fid_net, to_images(X_fid)))

    return {"test_mmd2": test_mmd2, "fid": fid}
