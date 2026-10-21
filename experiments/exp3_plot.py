"""Figures for Experiment 3 (UCI Concrete) from results/exp3_results.json.

Called at the end of exp3_concrete.py; can also be run on its own to redraw
the figures without retraining:

    python experiments/exp3_plot.py

Outputs:
  figures/exp3_mse_vs_epsilon.pdf      test MSE vs epsilon (Figure 2, left)
  figures/exp3_mmd_vs_epsilon.pdf      test MMD^2 vs epsilon (Figure 2, right)
  figures/exp3_pool_distribution.pdf   target vs contaminating Y in the training pool (Appendix)
  figures/exp3_scatter.pdf             true vs predicted Y at epsilon = 0.5 (Appendix)
  figures/exp3_qq.pdf                  predicted vs true Y quantiles at epsilon = 0.5 (Appendix)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

METHODS = ["oracle", "audit_only", "mse", "huber", "tukey", "naive_mmd", "cc_mmd"]

METHOD_STYLES = {
    "oracle":     dict(color="black",   ls="-",  marker="o", label="Oracle (MSE)"),
    "audit_only": dict(color="#17becf", ls="-.", marker="X", label="Audit-only"),
    "mse":        dict(color="#1f77b4", ls="--", marker="s", label="MSE"),
    "huber":      dict(color="#ff7f0e", ls="--", marker="^", label="Huber"),
    "tukey":      dict(color="#2ca02c", ls=":",  marker="D", label="Tukey"),
    "naive_mmd":  dict(color="#d62728", ls="--", marker="v", label="Naive MMD"),
    "cc_mmd":     dict(color="#7b1fa2", ls="-",  marker="*", label="CC-MMD"),
}


def load_results(path: str | Path) -> dict:
    """Load results/exp3_results.json; sweep keys become floats and arrays become numpy."""
    with open(path) as f:
        raw = json.load(f)
    demo = raw["demo"]
    for field in ("I_true", "Y_pool"):
        demo[field] = np.array(demo[field], dtype=np.float32)
    demo["preds"] = {k: np.array(v, dtype=np.float32) for k, v in demo["preds"].items()}
    return dict(
        cfg=raw["cfg"],
        sweep={float(eps): reps for eps, reps in raw["sweep"].items()},
        demo=demo,
        Y_test=np.array(raw["Y_test"], dtype=np.float32),
    )


def _plot_sweep(sweep: dict, metric: str, ylabel: str, legend_kw: dict, out_path: Path) -> None:
    """Mean +/- std over replicates of a test metric against epsilon, one line per method."""
    epsilons = sorted(sweep.keys())

    fig, ax = plt.subplots(figsize=(13, 8))
    for key in METHODS:
        st = METHOD_STYLES[key]
        means = np.array([np.mean([r[metric][key] for r in sweep[eps]]) for eps in epsilons])
        stds = np.array([np.std([r[metric][key] for r in sweep[eps]]) for eps in epsilons])
        ax.fill_between(epsilons, means - stds, means + stds, color=st["color"], alpha=0.15, zorder=1)
        ax.plot(epsilons, means, color=st["color"], linestyle=st["ls"], marker="o",
                linewidth=5.0, markersize=16, zorder=2, label=st["label"])

    ax.set_xlabel(r"Contamination rate $\varepsilon$", fontsize=36)
    ax.set_ylabel(ylabel, fontsize=36, labelpad=12)
    ax.set_xticks(epsilons)
    ax.set_xticklabels([f"{e:.1f}" for e in epsilons])
    ax.tick_params(axis="x", labelsize=33, pad=18)
    ax.tick_params(axis="y", labelsize=33)
    ax.legend(**legend_kw)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5, zorder=0)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22, left=0.18)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_mse_vs_epsilon(sweep: dict, out_path: Path) -> None:
    """Figure 2 (left)."""
    _plot_sweep(sweep, "mse", "Test MSE $\\downarrow$", dict(fontsize=22, loc="upper left"), out_path)


def plot_mmd_vs_epsilon(sweep: dict, out_path: Path) -> None:
    """Figure 2 (right)."""
    _plot_sweep(sweep, "mmd", r"Test MMD² $\downarrow$", dict(fontsize=26), out_path)


def plot_pool_distribution(demo: dict, out_path: Path, n_bins: int = 30) -> None:
    """Histogram of the training-pool Y values, split into target and contaminating points."""
    Y_pool = demo["Y_pool"]
    inlier_mask = demo["I_true"].astype(bool)
    contam_mask = ~inlier_mask
    bins = np.linspace(Y_pool.min() - 0.1, Y_pool.max() + 0.1, n_bins + 1)

    fig, ax = plt.subplots(figsize=(13, 8))
    ax.hist(Y_pool[inlier_mask], bins=bins, alpha=0.65, color="#1f77b4", label="Target (q)", density=True)
    if contam_mask.any():
        ax.hist(Y_pool[contam_mask], bins=bins, alpha=0.65, color="#ff7f0e",
                label="Contaminating (z)", density=True)
    ax.set_xlabel("$Y$ (standardized)", fontsize=36)
    ax.set_ylabel("Density", fontsize=36, labelpad=12)
    ax.tick_params(axis="x", labelsize=33, pad=18)
    ax.tick_params(axis="y", labelsize=33)
    ax.legend(fontsize=26)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22, left=0.18)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def _method_panels(n_methods: int, ncols: int = 3, panel_size: tuple = (5.5, 5)):
    nrows = int(np.ceil(n_methods / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_size[0] * ncols, panel_size[1] * nrows),
                             sharex=True, sharey=True, squeeze=False)
    return fig, axes.flatten(), nrows, ncols


def plot_scatter(demo: dict, Y_test: np.ndarray, out_path: Path) -> None:
    """True vs predicted test Y, one panel per method, with test MSE and MMD^2."""
    preds = demo["preds"]
    lo = min(Y_test.min(), min(preds[k].min() for k in METHODS)) - 0.5
    hi = max(Y_test.max(), max(preds[k].max() for k in METHODS)) + 0.5

    fig, ax_flat, nrows, ncols = _method_panels(len(METHODS))
    for ax, key in zip(ax_flat, METHODS):
        st = METHOD_STYLES[key]
        yhat = preds[key]
        mse = float(((yhat - Y_test) ** 2).mean())
        mmd = demo["mmd"][key]
        ax.scatter(Y_test, yhat, s=100, alpha=0.70, color=st["color"], marker=st["marker"], zorder=3)
        ax.plot([lo, hi], [lo, hi], "k--", lw=3.0, alpha=0.5, zorder=2)
        ax.set_title(st["label"], fontsize=28, pad=40)
        ax.text(0.5, 1.0, f"MSE={mse:.3f},  MMD²={mmd:.4f}",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=20)
        ax.set_aspect("equal")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.tick_params(labelsize=20, width=2.0, length=7)

    for ax in ax_flat[len(METHODS):]:
        ax.set_visible(False)
    for ax in ax_flat[(nrows - 1) * ncols: len(METHODS)]:
        ax.set_xlabel("True $Y$ (standardised)", fontsize=22)
    for ax in ax_flat[::ncols]:
        ax.set_ylabel(r"Predicted $\hat{Y}$", fontsize=22)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_qq(demo: dict, Y_test: np.ndarray, out_path: Path) -> None:
    """Sorted predicted vs sorted true test Y, one panel per method, with MMD^2 and R^2."""
    preds = demo["preds"]
    q_true = np.sort(Y_test)
    lo, hi = q_true.min() - 0.3, q_true.max() + 0.3

    fig, ax_flat, nrows, ncols = _method_panels(len(METHODS))
    for ax, key in zip(ax_flat, METHODS):
        st = METHOD_STYLES[key]
        q_pred = np.sort(preds[key])
        mmd = demo["mmd"][key]
        ax.scatter(q_true, q_pred, s=100, alpha=0.70, color=st["color"], marker=st["marker"], zorder=3)
        ax.plot([lo, hi], [lo, hi], "k--", lw=3.0, alpha=0.5, zorder=2)
        ax.set_title(st["label"], fontsize=28, pad=40)
        ax.text(0.5, 1.0, f"MMD²={mmd:.4f},  R²={r2_score(q_true, q_pred):.3f}",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=20)
        ax.set_aspect("equal")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.tick_params(labelsize=20, width=2.0, length=7)

    for ax in ax_flat[len(METHODS):]:
        ax.set_visible(False)
    for ax in ax_flat[(nrows - 1) * ncols: len(METHODS)]:
        ax.set_xlabel("True $Y$ quantiles (standardised)", fontsize=22)
    for ax in ax_flat[::ncols]:
        ax.set_ylabel(r"Predicted $\hat{Y}$ quantiles", fontsize=22)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_all(results: dict, out_dir: Path) -> None:
    """Draw the five Experiment 3 figures."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_mse_vs_epsilon(results["sweep"], out_dir / "exp3_mse_vs_epsilon.pdf")
    plot_mmd_vs_epsilon(results["sweep"], out_dir / "exp3_mmd_vs_epsilon.pdf")
    plot_pool_distribution(results["demo"], out_dir / "exp3_pool_distribution.pdf")
    plot_scatter(results["demo"], results["Y_test"], out_dir / "exp3_scatter.pdf")
    plot_qq(results["demo"], results["Y_test"], out_dir / "exp3_qq.pdf")


def main() -> None:
    p = argparse.ArgumentParser(description="Draw the Experiment 3 figures from saved results.")
    p.add_argument("--results", default="results/exp3_results.json")
    p.add_argument("--figures-dir", default="figures")
    args = p.parse_args()
    plot_all(load_results(args.results), Path(args.figures_dir))


if __name__ == "__main__":
    main()
