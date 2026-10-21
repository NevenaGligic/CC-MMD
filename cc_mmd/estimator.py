"""MMD^2 estimators for the contaminated two-sample problem (Section 2).

Notation follows the paper:
  X          clean reference sample from p, shape (n, d)
  W          pooled sample from r = (1 - eps) q + eps z, shape (N, d)
  I          true membership indicators (1 = drawn from q), shape (N,)
  I_tilde    proxy membership scores in [0, 1], shape (N,)
  L_indices  indices of the audited subset L, shape (m,)
  I_true_L   true membership indicators of the audited points, shape (m,)
"""

from typing import Callable

import torch
from torch import Tensor


def _mmd2_from_kernel_matrices(K_pp: Tensor, K_qq: Tensor, K_pq: Tensor) -> Tensor:
    """Unbiased U-statistic MMD^2 from full kernel matrices (diagonals excluded)."""
    n = K_pp.shape[0]
    N = K_qq.shape[0]

    K_pp_off = K_pp - torch.diag(torch.diag(K_pp))
    K_qq_off = K_qq - torch.diag(torch.diag(K_qq))

    A_pp, B_pp = K_pp_off.sum(), n * (n - 1)
    A_qq, B_qq = K_qq_off.sum(), N * (N - 1)
    A_pq, B_pq = K_pq.sum(), n * N

    return A_pp / B_pp + A_qq / B_qq - 2.0 * A_pq / B_pq


def oracle_mmd2(
    X: Tensor,
    W: Tensor,
    I: Tensor,
    kernel_fn: Callable[[Tensor, Tensor], Tensor],
) -> Tensor:
    """Oracle MMD^2: the U-statistic between X and the genuine pooled points (I_j = 1).

    Args:
        X: Reference sample, shape (n, d).
        W: Pooled sample, shape (N, d).
        I: True membership indicators in {0, 1}, shape (N,).
        kernel_fn: Callable (A, B) -> kernel matrix of shape (|A|, |B|).

    Returns:
        Scalar MMD^2 estimate (NaN if fewer than two genuine points).
    """
    Y = W[I.bool()]
    if Y.shape[0] < 2:
        return torch.tensor(float("nan"), device=X.device)
    return _mmd2_from_kernel_matrices(kernel_fn(X, X), kernel_fn(Y, Y), kernel_fn(X, Y))


def naive_mmd2(
    X: Tensor,
    W: Tensor,
    kernel_fn: Callable[[Tensor, Tensor], Tensor],
) -> Tensor:
    """Naive MMD^2: treats the whole pooled sample as if it were drawn from q.

    Args:
        X: Reference sample, shape (n, d).
        W: Pooled sample, shape (N, d).
        kernel_fn: Callable (A, B) -> kernel matrix of shape (|A|, |B|).

    Returns:
        Scalar MMD^2 estimate.
    """
    return _mmd2_from_kernel_matrices(kernel_fn(X, X), kernel_fn(W, W), kernel_fn(X, W))


def audit_only_mmd2(
    X: Tensor,
    W: Tensor,
    L_indices: Tensor,
    I_true_L: Tensor,
    kernel_fn: Callable[[Tensor, Tensor], Tensor],
) -> Tensor:
    """Audit-only MMD^2: the U-statistic between X and the audited genuine points.

    Args:
        X: Reference sample, shape (n, d).
        W: Pooled sample, shape (N, d).
        L_indices: Indices of the audited points, shape (m,).
        I_true_L: True membership indicators of the audited points, shape (m,).
        kernel_fn: Callable (A, B) -> kernel matrix of shape (|A|, |B|).

    Returns:
        Scalar MMD^2 estimate (NaN if fewer than two audited genuine points).
    """
    Y = W[L_indices[I_true_L.bool()]]
    if Y.shape[0] < 2:
        return torch.tensor(float("nan"), device=X.device)
    return _mmd2_from_kernel_matrices(kernel_fn(X, X), kernel_fn(Y, Y), kernel_fn(X, Y))


def cc_mmd2(
    X: Tensor,
    W: Tensor,
    I_tilde: Tensor,
    L_indices: Tensor,
    I_true_L: Tensor,
    kernel_fn: Callable[[Tensor, Tensor], Tensor],
) -> Tensor:
    """Contamination-corrected MMD^2 (CC-MMD, Section 2 and Algorithm 1).

    The qq and pq kernel sums and their denominators are estimated with the
    Horvitz-Thompson correction built from the audit residuals
    Delta_j = I_j - I_tilde_j (j in L), with Delta_j = 0 outside L.

    Args:
        X: Reference sample, shape (n, d).
        W: Pooled sample, shape (N, d).
        I_tilde: Proxy membership scores in [0, 1], shape (N,).
        L_indices: Indices of the audited subset, shape (m,).
        I_true_L: True membership indicators of the audited points, shape (m,).
        kernel_fn: Callable (A, B) -> kernel matrix of shape (|A|, |B|).

    Returns:
        Scalar CC-MMD^2 estimate (NaN if a corrected denominator is not positive).
    """
    n = X.shape[0]
    N = W.shape[0]
    m = L_indices.shape[0]

    Delta = torch.zeros(N, dtype=I_tilde.dtype, device=W.device)
    Delta[L_indices] = I_true_L.to(I_tilde.dtype) - I_tilde[L_indices]
    Delta_L = Delta[L_indices]

    # pp term: fully observed.
    K_pp = kernel_fn(X, X)
    A_pp_star = (K_pp - torch.diag(torch.diag(K_pp))).sum()
    B_pp_star = n * (n - 1)

    # qq term: A_hat_qq = proxy term + two single-sum corrections + pair correction.
    K_WW = kernel_fn(W, W)
    K_WW_off = K_WW - torch.diag(torch.diag(K_WW))
    outer_tilde = I_tilde.unsqueeze(1) * I_tilde.unsqueeze(0)

    T1_qq = (outer_tilde * K_WW_off).sum()
    T2_qq = (N / m) * (Delta_L * (K_WW_off[L_indices, :] @ I_tilde)).sum()
    T3_qq = T2_qq  # the two single-sum corrections are equal because k is symmetric
    if m >= 2:
        outer_delta = Delta_L.unsqueeze(1) * Delta_L.unsqueeze(0)
        K_LL = K_WW_off[L_indices][:, L_indices]
        T4_qq = (N * (N - 1) / (m * (m - 1))) * (outer_delta * K_LL).sum()
    else:
        T4_qq = torch.tensor(0.0, device=W.device)
    A_hat_qq = T1_qq + T2_qq + T3_qq + T4_qq

    # B_hat_qq: the same construction with k = 1 off the diagonal.
    ones_off = torch.ones(N, N, device=W.device) - torch.eye(N, device=W.device)
    T1_B = (outer_tilde * ones_off).sum()
    T2_B = (N / m) * (Delta_L * (ones_off[L_indices, :] @ I_tilde)).sum()
    T3_B = T2_B
    if m >= 2:
        T4_B = (N * (N - 1) / (m * (m - 1))) * (outer_delta * ones_off[L_indices][:, L_indices]).sum()
    else:
        T4_B = torch.tensor(0.0, device=W.device)
    B_hat_qq = T1_B + T2_B + T3_B + T4_B

    # pq term: proxy term + single-sum correction.
    K_XW = kernel_fn(X, W)
    A_hat_pq = (K_XW @ I_tilde).sum() + (N / m) * (K_XW[:, L_indices] @ Delta_L).sum()
    B_hat_pq = n * (I_tilde.sum() + (N / m) * Delta_L.sum())

    if B_hat_qq.item() <= 0 or B_hat_pq.item() <= 0:
        return torch.tensor(float("nan"), device=X.device)

    return A_pp_star / B_pp_star + A_hat_qq / B_hat_qq - 2.0 * A_hat_pq / B_hat_pq
