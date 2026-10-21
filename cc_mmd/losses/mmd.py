"""MMD^2 training losses for a fixed pool W (Experiments 3 and 4).

NaiveMMDLoss and CCMMDLoss compute the same quantities as naive_mmd2 and
cc_mmd2 in cc_mmd.estimator, with the generated samples X in the role of the
reference sample. Everything that depends only on W (including the N x N
kernel matrix) is computed once at construction; each call computes only the
X-X and X-W kernel terms.
"""

from __future__ import annotations

import torch
from torch import Tensor

from cc_mmd.kernels import gaussian_kernel


def median_bandwidth(X: Tensor, max_samples: int = 2000) -> float:
    """Median heuristic: sigma = median{||x_i - x_j|| : i < j}.

    Args:
        X: Data, shape (N, d).
        max_samples: If N exceeds this, the median is taken over a random subsample.

    Returns:
        Bandwidth sigma (at least 1e-6).
    """
    if X.shape[0] > max_samples:
        idx = torch.randperm(X.shape[0], device=X.device)[:max_samples]
        X = X[idx]
    with torch.no_grad():
        upper = torch.triu(torch.cdist(X, X, p=2), diagonal=1)
        sigma = upper[upper > 0].median().item()
    return max(sigma, 1e-6)


class NaiveMMDLoss:
    """MMD^2 between generated samples X and the whole pool W.

    Args:
        W: Fixed pool, shape (N, d).
        sigma: Gaussian kernel bandwidth.
    """

    def __init__(self, W: Tensor, sigma: float) -> None:
        N = W.shape[0]
        self._W = W
        self._N = N
        self._sigma = sigma

        with torch.no_grad():
            K_WW = gaussian_kernel(W, W, sigma)
            self._A_WW = (K_WW - torch.diag(torch.diag(K_WW))).sum()
            self._B_WW = float(N * (N - 1))

    def __call__(self, X: Tensor) -> Tensor:
        """MMD^2 for generated samples X of shape (n, d); differentiable in X."""
        n = X.shape[0]

        K_XX     = gaussian_kernel(X, X, self._sigma)
        K_XX_off = K_XX - torch.diag(torch.diag(K_XX))
        K_XW     = gaussian_kernel(X, self._W, self._sigma)

        A_XX = K_XX_off.sum()
        B_XX = n * (n - 1)
        A_XW = K_XW.sum()
        B_XW = n * self._N

        return A_XX / B_XX + self._A_WW / self._B_WW - 2.0 * A_XW / B_XW


class CCMMDLoss:
    """CC-MMD^2 between generated samples X and the latent clean part of the pool W.

    The corrected qq sums are computed once at construction. For the pq sums,
    the W-side weights (I_tilde and the audit residuals Delta) are stored and
    contracted with K(X, W) on each call.

    Args:
        W: Fixed contaminated pool, shape (N, d).
        I_tilde: Proxy membership scores in [0, 1], shape (N,).
        L_indices: Indices of the audited subset, shape (m,).
        I_true_L: True membership indicators of the audited points, shape (m,).
        sigma: Gaussian kernel bandwidth.
    """

    def __init__(
        self,
        W: Tensor,
        I_tilde: Tensor,
        L_indices: Tensor,
        I_true_L: Tensor,
        sigma: float,
    ) -> None:
        N = W.shape[0]
        m = L_indices.shape[0]
        self._W = W
        self._N = N
        self._m = m
        self._sigma = sigma
        self._L_indices = L_indices

        Delta = torch.zeros(N, dtype=I_tilde.dtype, device=W.device)
        Delta[L_indices] = I_true_L.to(I_tilde.dtype) - I_tilde[L_indices]
        Delta_L = Delta[L_indices]

        with torch.no_grad():
            K_WW = gaussian_kernel(W, W, sigma)
            K_WW_off = K_WW - torch.diag(torch.diag(K_WW))

            # ------------------------------------------------------------------
            # qq numerator  A_hat_qq — Eq. (4), four T terms
            # ------------------------------------------------------------------

            # T1: sum_{j != j'} I_tilde_j * I_tilde_{j'} * k(W_j, W_{j'})
            outer_tilde = I_tilde.unsqueeze(1) * I_tilde.unsqueeze(0)  # (N, N)
            T1_qq = (outer_tilde * K_WW_off).sum()

            # T2: (N/m) * sum_{j in L, j' != j} Delta_j * I_tilde_{j'} * k(W_j, W_{j'})
            K_L_all = K_WW_off[L_indices, :]   # (m, N)
            T2_qq = (N / m) * (Delta_L * (K_L_all @ I_tilde)).sum()

            # T3: symmetric to T2 (j and j' swapped) — equals T2 by symmetry of k
            T3_qq = T2_qq

            # T4: (N(N-1)/m(m-1)) * sum_{j,j' in L, j!=j'} Delta_j * Delta_{j'} * k(W_j, W_{j'})
            if m >= 2:
                K_LL        = K_WW_off[L_indices][:, L_indices]              # (m, m)
                outer_delta = Delta_L.unsqueeze(1) * Delta_L.unsqueeze(0)   # (m, m)
                T4_qq = (N * (N - 1) / (m * (m - 1))) * (outer_delta * K_LL).sum()
            else:
                T4_qq = torch.tensor(0.0, device=W.device)
            self._A_hat_qq = T1_qq + T2_qq + T3_qq + T4_qq

            # ------------------------------------------------------------------
            # qq denominator  B_hat_qq — same formula with k ≡ 1
            # ------------------------------------------------------------------
            ones_off = torch.ones(N, N, device=W.device) - torch.eye(N, device=W.device)
            T1_B = (outer_tilde * ones_off).sum()
            T2_B = (N / m) * (Delta_L * (ones_off[L_indices, :] @ I_tilde)).sum()
            T3_B = T2_B
            if m >= 2:
                ones_LL = ones_off[L_indices][:, L_indices]
                T4_B = (N * (N - 1) / (m * (m - 1))) * (outer_delta * ones_LL).sum()
            else:
                T4_B = torch.tensor(0.0, device=W.device)
            self._B_hat_qq = T1_B + T2_B + T3_B + T4_B

            # ------------------------------------------------------------------
            # pq denominator constant  (multiplied by n_gen each call)
            # B_hat_pq = n * (sum_j I_tilde_j + (N/m) * sum_{j in L} Delta_j)
            # ------------------------------------------------------------------
            self._pq_denom_scale = I_tilde.sum() + (N / m) * Delta_L.sum()

        self._I_tilde = I_tilde
        self._Delta_L = Delta_L

    def __call__(self, X: Tensor) -> Tensor:
        """CC-MMD^2 for generated samples X of shape (n, d); differentiable in X."""
        n = X.shape[0]

        # ------------------------------------------------------------------
        # pp term — fully observed, no correction needed
        # ------------------------------------------------------------------
        K_pp     = gaussian_kernel(X, X, self._sigma)
        K_pp_off = K_pp - torch.diag(torch.diag(K_pp))
        A_pp     = K_pp_off.sum()
        B_pp     = n * (n - 1)

        # ------------------------------------------------------------------
        # pq term — X-W cross kernel; W-side contracted with precomputed
        # I_tilde and Delta_L  (Eq. (5)-(6))
        # ------------------------------------------------------------------
        K_XW = gaussian_kernel(X, self._W, self._sigma)   # (n, N)

        # T1: sum_i sum_j I_tilde_j * k(X_i, W_j)
        A_pq_T1 = (K_XW @ self._I_tilde).sum()

        # T2: (N/m) * sum_i sum_{j in L} Delta_j * k(X_i, W_j)
        K_XL    = K_XW[:, self._L_indices]               # (n, m)
        A_pq_T2 = (self._N / self._m) * (K_XL @ self._Delta_L).sum()

        A_hat_pq = A_pq_T1 + A_pq_T2
        B_hat_pq = n * self._pq_denom_scale

        if self._B_hat_qq.item() <= 0 or B_hat_pq.item() <= 0:
            return torch.tensor(float("nan"), device=X.device)

        return A_pp / B_pp + self._A_hat_qq / self._B_hat_qq - 2.0 * A_hat_pq / B_hat_pq
