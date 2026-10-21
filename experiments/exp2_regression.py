"""Experiment 2: contaminated linear regression (Table 1, Appendix).

Fits Y = beta0 + beta1 X under 40% clustered contamination with MSE, Huber,
Tukey, naive MMD, audit-only MMD and CC-MMD.

Outputs:
  figures/regression_summary_200.pdf    estimates over 200 realizations, seeds 42-241 (Appendix)
  figures/regression_realizations.pdf   15 realizations, seeds 42, 1042, ..., 14042 (Appendix)
  results/exp2_summary_200_table.txt    Table 1 (mean +/- std over the 200 realizations)
  results/exp2_summary_200_table.tex    Table 1 in LaTeX
  results/exp2_summary_200_raw.json     per-realization estimates and proxy AUROC
  results/exp2_results.pt               data and fits of the 15 realizations

Usage:
    python experiments/exp2_regression.py
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import torch
from matplotlib.lines import Line2D
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cc_mmd.data import sample_audit_subset
from cc_mmd.data.synthetic import compute_proxy_scores_regression, generate_regression_data
from cc_mmd.estimator import audit_only_mmd2, cc_mmd2, naive_mmd2
from cc_mmd.kernels import gaussian_kernel

CFG = dict(
    N=300,
    epsilon=0.40,
    beta_true=(1.0, 2.0),
    sigma=0.5,
    K_clusters=4,
    m=40,
    sigma_k=1.0,
    n_model=10_000,
)

N_SUMMARY_RUNS = 200   # realizations summarized in Table 1 and regression_summary_200.pdf


# ---------------------------------------------------------------------------
# Pointwise estimators
# ---------------------------------------------------------------------------

def fit_ols(X_data: np.ndarray, Y_data: np.ndarray) -> Tuple[float, float]:
    """Ordinary least squares."""
    A = np.column_stack([np.ones_like(X_data), X_data])
    beta, _, _, _ = np.linalg.lstsq(A, Y_data, rcond=None)
    return float(beta[0]), float(beta[1])


def fit_huber(X_data: np.ndarray, Y_data: np.ndarray, epsilon: float = 1.25) -> Tuple[float, float]:
    """Huber regression with threshold delta = epsilon."""
    model = HuberRegressor(epsilon=epsilon, max_iter=200)
    model.fit(X_data.reshape(-1, 1), Y_data)
    return float(model.intercept_), float(model.coef_[0])


def _tukey_weights(residuals: np.ndarray, scale: float, c: float = 4.685) -> np.ndarray:
    """Tukey biweight IRLS weights."""
    u = residuals / (scale * c)
    return np.where(np.abs(u) <= 1.0, (1 - u ** 2) ** 2, 0.0)


def fit_tukey(X_data: np.ndarray, Y_data: np.ndarray,
              c: float = 4.685, max_iter: int = 50) -> Tuple[float, float]:
    """Tukey biweight regression by IRLS with a MAD scale, initialised at OLS."""
    beta0, beta1 = fit_ols(X_data, Y_data)
    A = np.column_stack([np.ones_like(X_data), X_data])
    for _ in range(max_iter):
        residuals = Y_data - (beta0 + beta1 * X_data)
        med = np.median(residuals)
        mad = np.median(np.abs(residuals - med))
        scale = 1.4826 * mad if mad > 0 else 1.0
        w = _tukey_weights(residuals, scale, c)
        if w.sum() < 2:
            break
        WA = A * w[:, None]
        try:
            beta, _, _, _ = np.linalg.lstsq(WA, Y_data * w, rcond=None)
        except np.linalg.LinAlgError:
            break
        beta0_new, beta1_new = float(beta[0]), float(beta[1])
        if abs(beta0_new - beta0) + abs(beta1_new - beta1) < 1e-8:
            break
        beta0, beta1 = beta0_new, beta1_new
    return beta0, beta1


# ---------------------------------------------------------------------------
# MMD-based estimators
# ---------------------------------------------------------------------------

def _make_model_samples(
    n_model: int,
    *,
    rng: torch.Generator,
    device: torch.device,
    sigma: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fixed covariates and noise for the model sample, shared across optimisation steps."""
    X_model = torch.rand(n_model, generator=rng, device=device) * 6.0 - 3.0
    eps_model = torch.randn(n_model, generator=rng, device=device) * sigma
    return X_model, eps_model


def _model_data_differentiable(
    beta_t: torch.Tensor,
    X_model: torch.Tensor,
    eps_model: torch.Tensor,
) -> torch.Tensor:
    """Model sample (X, beta0 + beta1 X + eps) with beta_t on the autograd graph."""
    Y_model = beta_t[0] + beta_t[1] * X_model + eps_model
    return torch.stack([X_model, Y_model], dim=1)


def _pooled_data_2d(X_data: torch.Tensor, Y_data: torch.Tensor) -> torch.Tensor:
    return torch.stack([X_data, Y_data], dim=1)


# Fixed starting points used, in addition to the OLS estimate, by every MMD-based fit.
_MULTISTART_GRID = [(b0, b1) for b0 in (-2.0, 0.0, 2.0) for b1 in (-2.0, 0.0, 2.0)]


def _minimise_multistart(objective_and_grad, beta_init: Tuple[float, float]) -> Tuple[float, float]:
    """Minimise an MMD-type objective with L-BFGS-B from several starting points.

    The starting points are beta_init (the OLS estimate on the pooled data) and the
    nine fixed points of _MULTISTART_GRID; the result with the lowest value of the
    objective being minimised is returned. The same procedure is used for naive MMD,
    audit-only MMD and CC-MMD, in every realization. MSE, Huber and Tukey are fitted
    with their own estimators.

    Several starts are needed because the empirical MMD^2 objective is flat for lines
    far from the data (all kernel entries underflow, so the gradient vanishes): a
    single run started from a strongly biased OLS estimate can stop there, far from
    the minimum.
    """
    best_x, best_f = None, np.inf
    for x0 in [tuple(beta_init)] + _MULTISTART_GRID:
        result = scipy.optimize.minimize(
            objective_and_grad, x0=np.array(x0, dtype=np.float64),
            method="L-BFGS-B", jac=True,
            options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-6},
        )
        if result.fun < best_f:
            best_x, best_f = result.x, float(result.fun)
    return float(best_x[0]), float(best_x[1])


def fit_naive_mmd(
    X_data: torch.Tensor,
    Y_data: torch.Tensor,
    cfg: dict,
    *,
    rng: torch.Generator,
    device: torch.device,
    beta_init: Tuple[float, float],
) -> Tuple[float, float]:
    """Minimise naive MMD^2 between the model sample and the whole pool over (beta0, beta1)."""
    kernel_fn = lambda A, B: gaussian_kernel(A, B, cfg["sigma_k"])
    W_2d = _pooled_data_2d(X_data, Y_data)
    X_model, eps_model = _make_model_samples(cfg["n_model"], rng=rng, device=device, sigma=cfg["sigma"])

    def objective_and_grad(params: np.ndarray):
        beta_t = torch.tensor(params, dtype=torch.float32, device=device, requires_grad=True)
        M = _model_data_differentiable(beta_t, X_model, eps_model)
        val = naive_mmd2(M, W_2d, kernel_fn)
        val.backward()
        grad = beta_t.grad.detach().cpu().numpy().astype(np.float64)
        return float(val.detach()), grad

    return _minimise_multistart(objective_and_grad, beta_init)


def fit_audit_only_mmd(
    X_data: torch.Tensor,
    Y_data: torch.Tensor,
    L_indices: torch.Tensor,
    I_true_L: torch.Tensor,
    cfg: dict,
    *,
    rng: torch.Generator,
    device: torch.device,
    beta_init: Tuple[float, float],
) -> Tuple[float, float]:
    """Minimise MMD^2 between the model sample and the audited genuine points only."""
    kernel_fn = lambda A, B: gaussian_kernel(A, B, cfg["sigma_k"])
    W_2d = _pooled_data_2d(X_data, Y_data)
    X_model, eps_model = _make_model_samples(cfg["n_model"], rng=rng, device=device, sigma=cfg["sigma"])

    def objective_and_grad(params: np.ndarray):
        beta_t = torch.tensor(params, dtype=torch.float32, device=device, requires_grad=True)
        M = _model_data_differentiable(beta_t, X_model, eps_model)
        val = audit_only_mmd2(M, W_2d, L_indices, I_true_L, kernel_fn)
        if not torch.isfinite(val):
            return 1e6, np.zeros(2)
        val.backward()
        grad = beta_t.grad.detach().cpu().numpy().astype(np.float64)
        return float(val.detach()), grad

    return _minimise_multistart(objective_and_grad, beta_init)


def fit_cc_mmd(
    X_data: torch.Tensor,
    Y_data: torch.Tensor,
    I_tilde: torch.Tensor,
    L_indices: torch.Tensor,
    I_true_L: torch.Tensor,
    cfg: dict,
    *,
    rng: torch.Generator,
    device: torch.device,
    beta_init: Tuple[float, float],
) -> Tuple[float, float]:
    """Minimise CC-MMD^2 between the model sample and the pool over (beta0, beta1)."""
    kernel_fn = lambda A, B: gaussian_kernel(A, B, cfg["sigma_k"])
    W_2d = _pooled_data_2d(X_data, Y_data)
    X_model, eps_model = _make_model_samples(cfg["n_model"], rng=rng, device=device, sigma=cfg["sigma"])

    def objective_and_grad(params: np.ndarray):
        beta_t = torch.tensor(params, dtype=torch.float32, device=device, requires_grad=True)
        M = _model_data_differentiable(beta_t, X_model, eps_model)
        val = cc_mmd2(M, W_2d, I_tilde, L_indices, I_true_L, kernel_fn)
        if not torch.isfinite(val):
            return 1e6, np.zeros(2)
        val.backward()
        grad = beta_t.grad.detach().cpu().numpy().astype(np.float64)
        return float(val.detach()), grad

    return _minimise_multistart(objective_and_grad, beta_init)


# ---------------------------------------------------------------------------
# One realization
# ---------------------------------------------------------------------------

def run_regression_realization(cfg: dict, seed: int, device: torch.device) -> dict:
    """Generate one contaminated dataset and fit all six estimators."""
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    X_data, Y_data, I, cluster_centers = generate_regression_data(
        cfg["N"], cfg["epsilon"], cfg["beta_true"], cfg["sigma"],
        K_clusters=cfg["K_clusters"], rng=rng, device=device,
    )
    I_tilde = compute_proxy_scores_regression(X_data, Y_data, rng=rng)
    L_indices = sample_audit_subset(cfg["N"], cfg["m"], rng=rng, device=device)
    I_true_L = I[L_indices]

    X_np = X_data.cpu().numpy()
    Y_np = Y_data.cpu().numpy()
    beta_ols = fit_ols(X_np, Y_np)
    beta_huber = fit_huber(X_np, Y_np)
    beta_tukey = fit_tukey(X_np, Y_np)

    rng_naive = torch.Generator(device=device)
    rng_naive.manual_seed(seed + 99_999)
    beta_naive = fit_naive_mmd(X_data, Y_data, cfg, rng=rng_naive, device=device, beta_init=beta_ols)

    rng_cc = torch.Generator(device=device)
    rng_cc.manual_seed(seed + 199_999)
    beta_cc = fit_cc_mmd(X_data, Y_data, I_tilde, L_indices, I_true_L, cfg,
                         rng=rng_cc, device=device, beta_init=beta_ols)

    rng_audit = torch.Generator(device=device)
    rng_audit.manual_seed(seed + 299_999)
    beta_audit = fit_audit_only_mmd(X_data, Y_data, L_indices, I_true_L, cfg,
                                    rng=rng_audit, device=device, beta_init=beta_ols)

    I_np = I.cpu().numpy()
    return dict(
        X_data=X_np,
        Y_data=Y_np,
        I=I_np,
        L_indices=L_indices.cpu().numpy(),
        cluster_centers=cluster_centers.cpu().numpy(),
        proxy_auroc=float(roc_auc_score(I_np, I_tilde.cpu().numpy())),
        betas=dict(
            ols=beta_ols,
            huber=beta_huber,
            tukey=beta_tukey,
            naive_mmd=beta_naive,
            audit_only=beta_audit,
            cc_mmd=beta_cc,
        ),
    )


def print_proxy_auroc(runs: list, label: str) -> None:
    """Print the proxy AUROC (mean +/- sample std) over a set of realizations."""
    auroc = np.array([d["proxy_auroc"] for d in runs])
    print(f"Proxy AUROC over the {label}: {auroc.mean():.4f} +/- {auroc.std(ddof=1):.4f}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_METHOD_STYLES = {
    "true":       dict(color="black",   ls="-",  label="True line"),
    "ols":        dict(color="#1f77b4", ls="--", label="MSE"),
    "huber":      dict(color="#ff7f0e", ls="--", label="Huber"),
    "tukey":      dict(color="#2ca02c", ls=":",  label="Tukey"),
    "naive_mmd":  dict(color="#d62728", ls="--", label="Naive MMD"),
    "audit_only": dict(color="#17becf", ls="-.", label="Audit-only"),
    # Long dashes let the true line show through where CC-MMD lies on top of it.
    "cc_mmd":     dict(color="#7b1fa2", ls=(0, (6, 2.5)), label="CC-MMD"),
}

# Fitted lines drawn in each panel, in legend order.
_FIT_KEYS = ["ols", "huber", "tukey", "naive_mmd", "audit_only", "cc_mmd"]

# (key, label) rows of Table 1.
_TABLE_ROWS = [
    ("ols",        "MSE"),
    ("huber",      "Huber"),
    ("tukey",      "Tukey"),
    ("naive_mmd",  "Naive MMD"),
    ("audit_only", "Audit-only"),
    ("cc_mmd",     "CC-MMD"),
]

_COLOR_INLIER = "#aec7e8"
_COLOR_CONTAM = "#ff7f0e"
_COLOR_AUDITED = "black"


def _plot_single_panel(ax: plt.Axes, data: dict, cfg: dict) -> None:
    """One realization: data, audited points, the true line and the six fits, with l2 errors as title."""
    X_np, Y_np = data["X_data"], data["Y_data"]
    I, L_idx, betas = data["I"], data["L_indices"], data["betas"]
    b0_true, b1_true = cfg["beta_true"]
    inliers = I == 1
    contams = ~inliers

    ax.scatter(X_np[contams], Y_np[contams], s=140, alpha=0.7, color=_COLOR_CONTAM,
               label="Contaminating", zorder=1)
    ax.scatter(X_np[inliers], Y_np[inliers], s=140, alpha=0.6, color=_COLOR_INLIER,
               label="Target", zorder=2)
    ax.scatter(X_np[L_idx], Y_np[L_idx], s=300, facecolors="none",
               edgecolors=_COLOR_AUDITED, linewidths=1.5, label="Audited", zorder=3)

    x_line = np.linspace(-3.5, 3.5, 100)
    st = _METHOD_STYLES["true"]
    ax.plot(x_line, b0_true + b1_true * x_line,
            color=st["color"], lw=5.5, ls=st["ls"], label=st["label"], zorder=5)

    errs = {}
    for key in _FIT_KEYS:
        b0, b1 = betas[key]
        st = _METHOD_STYLES[key]
        ax.plot(x_line, b0 + b1 * x_line, zorder=6 if key == "cc_mmd" else 4,
                color=st["color"], lw=4.5, ls=st["ls"], label=st["label"])
        errs[key] = np.sqrt((b0 - b0_true) ** 2 + (b1 - b1_true) ** 2)

    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-10, 14)
    ax.tick_params(axis="both", labelsize=24, width=2.5, length=8)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5, zorder=0)

    err_t1 = [f"MSE: {errs['ols']:.2f}", f"Huber: {errs['huber']:.2f}",
              f"Tukey: {errs['tukey']:.2f}"]
    err_t2 = [f"Naive: {errs['naive_mmd']:.2f}", f"Audit: {errs['audit_only']:.2f}",
              f"CC-MMD: {errs['cc_mmd']:.2f}"]
    ax.set_title(" | ".join(err_t1) + "\n" + " | ".join(err_t2),
                 fontsize=28, pad=15, fontweight="normal")


def plot_regression_realizations(all_data: list, cfg: dict, output_path: Path, n_cols: int = 3) -> None:
    """Grid of realizations with a shared legend (Appendix)."""
    n = len(all_data)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9 * n_cols, 8 * n_rows),
                             sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for i, (ax, data) in enumerate(zip(axes_flat, all_data)):
        _plot_single_panel(ax, data, cfg)
        if i >= (n_rows - 1) * n_cols:
            ax.set_xlabel("$X$", fontsize=28, fontweight="bold")
        if i % n_cols == 0:
            ax.set_ylabel("$Y$", fontsize=28, fontweight="bold", labelpad=0)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    # Legend in two rows: data and true line (padded with two blanks), then the six fits.
    h, lb = axes_flat[0].get_legend_handles_labels()
    spacers = [Line2D([], [], color="none") for _ in range(2)]
    handles = [h[1], h[0], h[2], h[3]] + spacers + h[4:]
    labels = [lb[1], lb[0], lb[2], lb[3], "", ""] + lb[4:]
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=6, fontsize=36, frameon=False)

    fig.tight_layout()
    fig.subplots_adjust(top=0.91)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


# ---------------------------------------------------------------------------
# Summary over many realizations (Table 1 and figure)
# ---------------------------------------------------------------------------

def _aggregate_runs(all_runs: list, cfg: dict) -> dict:
    """Per-method arrays of intercept, slope and l2 error over all realizations."""
    b0_true, b1_true = cfg["beta_true"]
    agg = {k: {"b0": [], "b1": [], "l2": []} for k in _FIT_KEYS}
    for data in all_runs:
        for key in _FIT_KEYS:
            b0, b1 = data["betas"][key]
            agg[key]["b0"].append(b0)
            agg[key]["b1"].append(b1)
            agg[key]["l2"].append(np.sqrt((b0 - b0_true) ** 2 + (b1 - b1_true) ** 2))
    for key in _FIT_KEYS:
        for f in ("b0", "b1", "l2"):
            agg[key][f] = np.asarray(agg[key][f], dtype=float)
    return agg


def plot_regression_summary(agg: dict, cfg: dict, n: int, output_path: Path) -> None:
    """Intercept, slope and l2 error of every realization, with mean +/- std per method."""
    b0_true, b1_true = cfg["beta_true"]

    fig, axes = plt.subplots(1, 3, figsize=(24, 8.5))
    rng_jit = np.random.default_rng(0)

    panels = [
        ("b0", r"Intercept $\beta_0$", b0_true),
        ("b1", r"Slope $\beta_1$",     b1_true),
        ("l2", r"$\ell_2$ error",      None),
    ]
    pt_alpha = 0.18 if n <= 50 else 0.10
    pt_size = 28 if n <= 50 else 16

    for ax, (field, ylabel, truth) in zip(axes, panels):
        for i, key in enumerate(_FIT_KEYS):
            vals = agg[key][field]
            color = _METHOD_STYLES[key]["color"]
            jit = rng_jit.normal(0.0, 0.07, size=vals.shape)
            ax.scatter(i + jit, vals, s=pt_size, color=color, alpha=pt_alpha,
                       edgecolors="none", zorder=2)
            ax.errorbar(i, vals.mean(), yerr=vals.std(ddof=1),
                        fmt="o", color=color, markersize=18,
                        markeredgecolor="black", markeredgewidth=2.0,
                        ecolor="black", elinewidth=3.0, capsize=10, capthick=3.0,
                        zorder=4)

        if truth is not None:
            ax.axhline(truth, color="black", lw=3.0, ls="--", zorder=3, label=f"True = {truth:g}")
            ax.legend(fontsize=22, loc="best", framealpha=0.7)

        ax.set_xticks(range(len(_FIT_KEYS)))
        ax.set_xticklabels([_METHOD_STYLES[k]["label"] for k in _FIT_KEYS],
                           fontsize=21, rotation=30, ha="right")
        for tick, k in zip(ax.get_xticklabels(), _FIT_KEYS):
            if k == "cc_mmd":
                tick.set_fontweight("bold")
        ax.set_ylabel(ylabel, fontsize=30, labelpad=10)
        ax.tick_params(axis="y", labelsize=24)
        ax.grid(axis="y", linewidth=0.4, alpha=0.5, zorder=0)
        ax.set_xlim(-0.6, len(_FIT_KEYS) - 0.4)

        # Log scale when the mean l2 errors span more than a factor of 20.
        if field == "l2":
            means = np.array([agg[k][field].mean() for k in _FIT_KEYS])
            if means.max() / max(means.min(), 1e-12) > 20.0:
                ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def save_summary_tables(agg: dict, cfg: dict, seeds: list, txt_path: Path, tex_path: Path) -> None:
    """Write Table 1 (mean +/- std over realizations) as plain text and as LaTeX."""
    b0_true, b1_true = cfg["beta_true"]
    n = len(seeds)

    lines = [f"Mean +/- standard deviation over {n} realizations (seeds {seeds[0]}-{seeds[-1]})",
             "",
             f"{'Method':<12}  {'Intercept':>16}  {'Slope':>16}  {'L2 Error':>16}",
             "-" * 66,
             f"{'True':<12}  {b0_true:>16.3f}  {b1_true:>16.3f}  {'--':>16}"]
    for key, label in _TABLE_ROWS:
        a = agg[key]
        lines.append(
            f"{label:<12}  {a['b0'].mean():>8.3f} +-{a['b0'].std(ddof=1):<6.3f}  "
            f"{a['b1'].mean():>8.3f} +-{a['b1'].std(ddof=1):<6.3f}  "
            f"{a['l2'].mean():>8.3f} +-{a['l2'].std(ddof=1):<6.3f}"
        )
    txt_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Saved {txt_path}")

    best = min(_TABLE_ROWS, key=lambda kv: agg[kv[0]]["l2"].mean())[0]
    tex = [
        r"\begin{tabular}{lccc}",
        r"  \toprule",
        r"  Method & Intercept & Slope & $\ell_2$ error \\",
        r"  \midrule",
        rf"  True & {b0_true:.3f} & {b1_true:.3f} & -- \\",
    ]
    for key, label in _TABLE_ROWS:
        a = agg[key]
        cells = [rf"${a[f].mean():.3f} \pm {a[f].std(ddof=1):.3f}$" for f in ("b0", "b1", "l2")]
        name = label.replace("CC-MMD", "CC--MMD").replace("Naive MMD", "Na\\\"ive MMD")
        if key == best:
            cells = [rf"$\mathbf{{{c[1:-1]}}}$" for c in cells]
        tex.append("  " + " & ".join([name] + cells) + r" \\")
    tex += [r"  \bottomrule", r"\end{tabular}"]
    tex_path.write_text("\n".join(tex) + "\n")
    print(f"Saved {tex_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 2: contaminated linear regression")
    parser.add_argument("--n-realizations", type=int, default=15,
                        help="Realizations shown in regression_realizations.pdf.")
    parser.add_argument("--n-model", type=int, default=None,
                        help=f"Model sample size of the MMD-based fits (default {CFG['n_model']}).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--figures-dir", type=str, default="figures")
    parser.add_argument("--results-dir", type=str, default="results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if args.n_model is not None:
        CFG["n_model"] = args.n_model
    output_dir = Path(args.figures_dir)
    results_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}  |  Seed: {args.seed}")

    # Table 1 and the summary figure: realizations with seeds seed, seed+1, ..., seed+199.
    seeds = list(range(args.seed, args.seed + N_SUMMARY_RUNS))
    print(f"\n[1/2] {N_SUMMARY_RUNS} realizations for Table 1 ...")
    all_runs = [run_regression_realization(CFG, seed=s, device=device)
                for s in tqdm(seeds, desc="Realizations")]

    raw_path = results_dir / "exp2_summary_200_raw.json"
    with open(raw_path, "w") as f:
        json.dump([{"seed": s, "proxy_auroc": d["proxy_auroc"],
                    "betas": {k: list(map(float, v)) for k, v in d["betas"].items()}}
                   for s, d in zip(seeds, all_runs)], f, indent=2)
    print(f"Saved {raw_path}")

    agg = _aggregate_runs(all_runs, CFG)
    plot_regression_summary(agg, CFG, len(all_runs), output_dir / "regression_summary_200.pdf")
    save_summary_tables(agg, CFG, seeds,
                        results_dir / "exp2_summary_200_table.txt",
                        results_dir / "exp2_summary_200_table.tex")
    print_proxy_auroc(all_runs, f"{N_SUMMARY_RUNS} realizations")

    # Realizations grid: seeds seed, seed+1000, ..., seed+1000*(n_realizations-1).
    print(f"\n[2/2] {args.n_realizations} realizations for the grid figure ...")
    all_data = [run_regression_realization(CFG, seed=args.seed + i * 1000, device=device)
                for i in tqdm(range(args.n_realizations), desc="Realizations")]
    plot_regression_realizations(all_data, CFG, output_dir / "regression_realizations.pdf")
    results_path = results_dir / "exp2_results.pt"
    torch.save({"all_data": all_data, "cfg": CFG}, results_path)
    print(f"Saved {results_path}")
    print_proxy_auroc(all_data, f"{args.n_realizations} grid realizations")

    print("\nExperiment 2 complete.")


if __name__ == "__main__":
    main()
