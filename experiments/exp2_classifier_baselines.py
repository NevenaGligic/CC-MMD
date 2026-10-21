"""Experiment 2, appendix: CC-MMD vs. classifier-based baselines (Tables 3 and 4).

The contaminated linear regression of Experiment 2 (40% clustered contamination,
true line Y = 1 + 2X), with inlier noise sigma = 1. Four estimators of the line
minimise MMD^2 between a model sample and (part of) the pool:

  Naive MMD          the whole pool, no correction
  CC-MMD             the whole pool, proxy scores corrected with the audited points
  RF prob.-weighted  the unaudited points, weighted by the clean probabilities of a
                     random forest fitted on the audited points
  RF hard-threshold  the unaudited points whose random-forest clean probability is
                     at least 0.5

Table 3 has no upstream classifier (I_tilde = 1); Table 4 uses the weak proxy
score of Experiment 2. Each table reports the slope MSE +/- SE over 50
realizations for every pool size N in {200, 300, 500, 1000} and audit size
m in {20, 40, 60}. The k-th (N, m) pair, k = 1, ..., 12 in that order, uses
seeds 1000 + 1000 k + r, r = 0, ..., 49.

Outputs:
  results/exp2_classifier_baselines_table{3,4}.txt        Tables 3 and 4
  results/exp2_classifier_baselines_table{3,4}.tex        Tables 3 and 4 in LaTeX
  results/exp2_classifier_baselines_table{3,4}_raw.json   per-realization estimates

Usage:
    python experiments/exp2_classifier_baselines.py              # Tables 3 and 4
    python experiments/exp2_classifier_baselines.py --tables 4   # Table 4 only
"""

import argparse
import json
import sys
from itertools import product
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import scipy.optimize
import torch
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cc_mmd.estimator import cc_mmd2, naive_mmd2
from cc_mmd.kernels import gaussian_kernel

# All MMD objectives are computed in double precision.
torch.set_default_dtype(torch.float64)

CFG = dict(
    epsilon=0.40,
    beta_true=(1.0, 2.0),
    sigma=1.0,            # inlier noise standard deviation
    K_clusters=4,
    n_model=800,          # size of the model sample in each MMD^2 objective
    sigma_k=1.0,          # kernel bandwidth
    rf_n_estimators=400,
    rf_threshold=0.50,
    N_values=[200, 300, 500, 1000],
    m_values=[20, 40, 60],
    n_realizations=50,
    base_seed=1000,
)

METHODS = [
    ("naive_mmd", "Naive MMD"),
    ("cc_mmd", "CC-MMD"),
    ("rf_weighted", "RF prob.-weighted"),
    ("rf_hard", "RF hard-threshold"),
]

TABLE_PROXY = {3: "none", 4: "weak"}
CC_MMD_TEX_LABEL = {3: r"CC--MMD ($\tilde I_j \equiv 1$)", 4: "CC--MMD (weak proxy)"}


# ---------------------------------------------------------------------------
# Data, proxy and random forest
# ---------------------------------------------------------------------------

def generate_data(rng: np.random.Generator, N: int, cfg: dict):
    """Contaminated regression data with the contamination model of Experiment 2.

    round(epsilon N) contaminants come from K_clusters Gaussian clusters with
    centres drawn from Uniform[-4.5, 4.5] x Uniform[-8, 8] (std 0.35 in x and
    0.75 in y); the other points follow Y = beta0 + beta1 X + N(0, sigma^2) with
    X ~ Uniform[-3, 3]. Returns the shuffled x, y and the membership indicators
    I (1 = inlier).
    """
    n_out = int(round(cfg["epsilon"] * N))
    n_in = N - n_out
    beta0, beta1 = cfg["beta_true"]

    x_in = rng.uniform(-3.0, 3.0, size=n_in)
    y_in = beta0 + beta1 * x_in + rng.normal(0.0, cfg["sigma"], size=n_in)

    centers_x = rng.uniform(-4.5, 4.5, size=cfg["K_clusters"])
    centers_y = rng.uniform(-8.0, 8.0, size=cfg["K_clusters"])
    cluster_ids = rng.integers(0, cfg["K_clusters"], size=n_out)
    x_out = centers_x[cluster_ids] + rng.normal(0.0, 0.35, size=n_out)
    y_out = centers_y[cluster_ids] + rng.normal(0.0, 0.75, size=n_out)

    order = rng.permutation(N)
    x = np.concatenate([x_in, x_out])[order]
    y = np.concatenate([y_in, y_out])[order]
    I = np.concatenate([np.ones(n_in, dtype=int), np.zeros(n_out, dtype=int)])[order]
    return x, y, I


def ols(x: np.ndarray, y: np.ndarray, weights: np.ndarray = None) -> np.ndarray:
    """(Weighted) least-squares estimate of (beta0, beta1)."""
    A = np.column_stack([np.ones_like(x), x])
    if weights is None:
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    else:
        sw = np.sqrt(np.clip(weights, 0.0, None) + 1e-12)
        beta, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
    return beta


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))


def weak_proxy_scores(rng: np.random.Generator, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Weak proxy of Experiment 2: sigmoid(1.5 - 1.2 z + N(0, 0.9^2)), where z is the
    absolute residual of the pooled OLS fit divided by its MAD scale."""
    beta = ols(x, y)
    residual = y - (beta[0] + beta[1] * x)
    robust_scale = 1.4826 * np.median(np.abs(residual - np.median(residual))) + 1e-8
    score = 1.5 - 1.2 * np.abs(residual) / robust_scale
    score += rng.normal(0.0, 0.9, size=len(x))
    return np.clip(_sigmoid(score), 1e-4, 1.0 - 1e-4)


def rf_clean_probabilities(x: np.ndarray, y: np.ndarray, I: np.ndarray, audit_idx: np.ndarray,
                           seed: int, n_estimators: int) -> np.ndarray:
    """Clean probabilities of every point from a random forest fitted on the audited points.

    Features: x, y, the pooled OLS residual, its absolute value and x^2.
    """
    beta = ols(x, y)
    residual = y - (beta[0] + beta[1] * x)
    features = np.column_stack([x, y, residual, np.abs(residual), x ** 2])
    labels = I[audit_idx]

    if np.unique(labels).size == 1:
        return np.clip(np.full(len(x), float(labels[0])), 1e-4, 1.0 - 1e-4)

    clf = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=3, class_weight="balanced",
                                 max_features="sqrt", random_state=seed)
    clf.fit(features[audit_idx], labels)
    p_clean = clf.predict_proba(features)[:, int(np.where(clf.classes_ == 1)[0][0])]
    return np.clip(p_clean, 1e-4, 1.0 - 1e-4)


# ---------------------------------------------------------------------------
# MMD fits
# ---------------------------------------------------------------------------

def weighted_mmd2(X: torch.Tensor, W: torch.Tensor, w: torch.Tensor, kernel_fn: Callable) -> torch.Tensor:
    """MMD^2 between X and the sample W with weights w (U-statistics, diagonals excluded)."""
    n = X.shape[0]
    K_XX = kernel_fn(X, X)
    K_WW = kernel_fn(W, W)
    A_pp = K_XX.sum() - K_XX.diagonal().sum()
    A_qq = w @ K_WW @ w - (K_WW.diagonal() * w * w).sum()
    B_qq = w.sum() ** 2 - (w * w).sum()
    A_pq = (kernel_fn(X, W) @ w).sum()
    return A_pp / (n * (n - 1)) + A_qq / B_qq - 2.0 * A_pq / (n * w.sum())


def fit_mmd(mmd_fn: Callable, x_model: torch.Tensor, eps_model: torch.Tensor,
            beta_init: np.ndarray) -> np.ndarray:
    """Minimise mmd_fn(model sample) over (beta0, beta1) with L-BFGS-B.

    The model sample is (x_model, beta0 + beta1 x_model + eps_model). The fit starts
    from beta_init and from the four points 0.5 away from it along each axis; the
    result with the lowest objective value is returned.
    """
    def objective_and_grad(params: np.ndarray):
        beta = torch.tensor(params, dtype=torch.float64, device=x_model.device, requires_grad=True)
        M = torch.stack([x_model, beta[0] + beta[1] * x_model + eps_model], dim=1)
        val = mmd_fn(M)
        if not torch.isfinite(val):
            return 1e6, np.zeros(2)
        val.backward()
        return float(val.detach()), beta.grad.cpu().numpy()

    beta_init = np.asarray(beta_init, dtype=float)
    starts = [beta_init] + [beta_init + d for d in ([0.5, 0.0], [-0.5, 0.0], [0.0, 0.5], [0.0, -0.5])]
    best = None
    for x0 in starts:
        result = scipy.optimize.minimize(objective_and_grad, x0, method="L-BFGS-B", jac=True,
                                         options={"maxiter": 250, "gtol": 1e-6})
        if np.isfinite(result.fun) and (best is None or result.fun < best.fun):
            best = result
    return best.x


# ---------------------------------------------------------------------------
# One realization
# ---------------------------------------------------------------------------

def run_realization(cfg: dict, N: int, m: int, seed: int, proxy: str,
                    device: torch.device) -> Dict[str, np.ndarray]:
    """Generate one dataset and fit the four estimators; returns {method: (beta0, beta1)}."""
    rng = np.random.default_rng(seed)
    x, y, I = generate_data(rng, N, cfg)
    audit_idx = np.sort(rng.choice(N, size=m, replace=False))
    I_tilde = weak_proxy_scores(rng, x, y) if proxy == "weak" else np.ones(N)
    p_clean = rf_clean_probabilities(x, y, I, audit_idx, seed, cfg["rf_n_estimators"])

    unaudited = np.ones(N, dtype=bool)
    unaudited[audit_idx] = False
    hard_keep = unaudited & (p_clean >= cfg["rf_threshold"])
    if hard_keep.sum() < 2:
        candidates = np.where(unaudited)[0]
        hard_keep[candidates[np.argsort(p_clean[candidates])[-2:]]] = True

    # One model sample, shared by the four fits.
    x_model = rng.uniform(-3.0, 3.0, size=cfg["n_model"])
    eps_model = rng.normal(0.0, cfg["sigma"], size=cfg["n_model"])

    to_t = lambda a: torch.as_tensor(a, dtype=torch.float64, device=device)
    kernel_fn = lambda A, B: gaussian_kernel(A, B, cfg["sigma_k"])
    W = to_t(np.column_stack([x, y]))
    x_model, eps_model = to_t(x_model), to_t(eps_model)
    L_indices = torch.as_tensor(audit_idx, device=device)
    beta_ols = ols(x, y)

    return dict(
        naive_mmd=fit_mmd(lambda M: naive_mmd2(M, W, kernel_fn), x_model, eps_model, beta_ols),
        cc_mmd=fit_mmd(lambda M: cc_mmd2(M, W, to_t(I_tilde), L_indices, to_t(I[audit_idx]), kernel_fn),
                       x_model, eps_model, beta_ols),
        rf_weighted=fit_mmd(lambda M: weighted_mmd2(M, W[unaudited], to_t(p_clean[unaudited]), kernel_fn),
                            x_model, eps_model, ols(x[unaudited], y[unaudited], weights=p_clean[unaudited])),
        rf_hard=fit_mmd(lambda M: naive_mmd2(M, W[hard_keep], kernel_fn),
                        x_model, eps_model, ols(x[hard_keep], y[hard_keep])),
    )


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def slope_mse_se(slopes: np.ndarray, beta1: float):
    """Slope MSE and its standard error under a normal approximation of the slope errors."""
    err = slopes - beta1
    mse, bias = np.mean(err ** 2), np.mean(err)
    var = max(mse - bias ** 2, 0.0)
    return mse, np.sqrt((2.0 * var ** 2 + 4.0 * bias ** 2 * var) / len(err))


def save_table(records: list, table: int, cfg: dict, txt_path: Path, tex_path: Path) -> None:
    """Write the slope MSE +/- SE table as plain text and as LaTeX (lowest MSE per row in bold)."""
    lines = [f"Table {table}: slope MSE +/- SE over {cfg['n_realizations']} realizations "
             f"({'no upstream classifier' if TABLE_PROXY[table] == 'none' else 'weak upstream classifier'})",
             "",
             f"{'N':>5} {'m':>3}" + "".join(f"  {label:>18}" for _, label in METHODS)]
    tex = [r"\begin{tabular}{llcccc}", r"  \toprule",
           r"  $N$ & $m$ & Na\"ive MMD & " + CC_MMD_TEX_LABEL[table] + r" & RF prob.-weighted & RF hard-threshold \\",
           r"  \midrule"]
    for N, m in product(cfg["N_values"], cfg["m_values"]):
        stats = []
        for key, _ in METHODS:
            slopes = np.array([r["betas"][key][1] for r in records if r["N"] == N and r["m"] == m])
            stats.append(slope_mse_se(slopes, cfg["beta_true"][1]))
        best = min(mse for mse, _ in stats)
        lines.append(f"{N:>5} {m:>3}" + "".join(f"  {f'{mse:.4f} +- {se:.4f}':>18}" for mse, se in stats))
        cells = [f"{mse:.4f} $\\pm$ {se:.4f}" for mse, se in stats]
        cells = [rf"\textbf{{{c}}}" if np.isclose(mse, best, rtol=1e-10, atol=1e-12) else c
                 for c, (mse, _) in zip(cells, stats)]
        tex.append(f"  {N} & {m} & " + " & ".join(cells) + r" \\")
    tex += [r"  \bottomrule", r"\end{tabular}"]

    txt_path.write_text("\n".join(lines) + "\n")
    tex_path.write_text("\n".join(tex) + "\n")
    print("\n".join(lines))
    print(f"Saved {txt_path}\nSaved {tex_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 2 appendix: Tables 3 and 4.")
    parser.add_argument("--tables", type=int, nargs="+", choices=[3, 4], default=[3, 4])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--results-dir", type=str, default="results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    for table in args.tables:
        proxy = TABLE_PROXY[table]
        records = []
        for k, (N, m) in enumerate(product(CFG["N_values"], CFG["m_values"]), start=1):
            for r in tqdm(range(CFG["n_realizations"]), desc=f"Table {table}: N = {N}, m = {m}"):
                seed = CFG["base_seed"] + 1000 * k + r
                betas = run_realization(CFG, N, m, seed, proxy, device)
                records.append(dict(N=N, m=m, seed=seed, betas={key: [float(b) for b in beta]
                                                                for key, beta in betas.items()}))

        stem = f"exp2_classifier_baselines_table{table}"
        with open(results_dir / f"{stem}_raw.json", "w") as f:
            json.dump(records, f, indent=1)
        print(f"Saved {results_dir / f'{stem}_raw.json'}")
        save_table(records, table, CFG, results_dir / f"{stem}.txt", results_dir / f"{stem}.tex")


if __name__ == "__main__":
    main()
