"""Figure 4: generative quality against the contamination rate (IAF, K-MNIST).

Reads results/exp4_ablation_epsilon_results.json (written by exp4_ablation_epsilon.py,
which also calls this script at the end).

Outputs:
  figures/exp4_ablation_epsilon_mmd2.pdf   test MMD^2 vs epsilon, log scale (Figure 4, left)
  figures/exp4_ablation_epsilon_fid.pdf    FID vs epsilon (Figure 4, right)

Usage:
    python experiments/exp4_plot_ablation_epsilon.py
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.ticker import LogLocator

MODEL_STYLE: Dict[str, dict] = {
    "oracle":     dict(color="black",   linestyle="-",  label="Oracle"),
    "audit_only": dict(color="#17becf", linestyle="--", label="Audit-only"),
    "naive_mmd":  dict(color="#d62728", linestyle="-",  label="Naïve MMD"),
    "swd":        dict(color="#e69c00", linestyle="-.", label="SWD"),
    "cc_mmd":     dict(color="#7b1fa2", linestyle="-",  label="CC-MMD"),
}
MODEL_ORDER = ["oracle", "audit_only", "naive_mmd", "swd", "cc_mmd"]


def aggregate(records: list[dict]):
    """Per model, the mean and std over replicates of FID and test MMD^2 at each epsilon."""
    groups: dict[float, list[dict]] = defaultdict(list)
    for r in records:
        groups[r["epsilon"]].append(r)
    epsilons = sorted(groups.keys())

    def stat(fn, key):
        return {name: np.array([fn([r[f"{name}_{key}"] for r in groups[e]]) for e in epsilons])
                for name in MODEL_ORDER}

    return epsilons, stat(np.mean, "fid"), stat(np.std, "fid"), stat(np.mean, "mmd2"), stat(np.std, "mmd2")


def _log_label(x: float, _) -> str:
    """Tick label 10^n or m x 10^n."""
    exp = int(np.floor(np.log10(x) + 1e-9))
    mantissa = round(x / 10 ** exp)
    return f"$10^{{{exp}}}$" if mantissa == 1 else f"${mantissa}\\times10^{{{exp}}}$"


def plot_metric(
    epsilons: list[float],
    mean_data: Dict[str, np.ndarray],
    std_data: Dict[str, np.ndarray],
    y_label: str,
    out_path: Path,
    figsize: tuple,
    log_y: bool = False,
) -> None:
    """Mean +/- std of each model against the contamination rate."""
    fig, ax = plt.subplots(figsize=figsize)
    eps_arr = np.array(epsilons)

    for name in MODEL_ORDER:
        sty = MODEL_STYLE[name]
        mean, std = mean_data[name], std_data[name]
        lo = np.maximum(mean - std, mean * 1e-3) if log_y else mean - std   # log axis needs positive values
        ax.fill_between(eps_arr, lo, mean + std, color=sty["color"], alpha=0.25, zorder=1)
        ax.plot(eps_arr, mean, color=sty["color"], linestyle=sty["linestyle"], marker="o",
                label=sty["label"], markersize=16, linewidth=5.0, zorder=2)

    ax.set_xlabel(r"Contamination rate $\varepsilon$", fontsize=36)
    ax.set_ylabel(y_label, fontsize=36, labelpad=12)
    x_ticks = [0.0, 0.2, 0.4, 0.6, 0.8]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{e:.1f}" for e in x_ticks])
    ax.tick_params(axis="x", labelsize=33, pad=18)
    ax.tick_params(axis="y", which="both", labelsize=33)
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    if log_y:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5), numticks=12))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_log_label))
        ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.legend(fontsize=26)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5, zorder=0)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22, left=0.18)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_all(records: list[dict], figures_dir: Path) -> None:
    """Draw both panels of Figure 4 (sized to have equal heights side by side)."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    epsilons, mean_fid, std_fid, mean_mmd2, std_mmd2 = aggregate(records)
    plot_metric(epsilons, mean_mmd2, std_mmd2, "Test MMD² ↓",
                figures_dir / "exp4_ablation_epsilon_mmd2.pdf", figsize=(13, 9), log_y=True)
    plot_metric(epsilons, mean_fid, std_fid, "FID ↓",
                figures_dir / "exp4_ablation_epsilon_fid.pdf", figsize=(14, 9))


def main() -> None:
    p = argparse.ArgumentParser(description="Figure 4 of the paper.")
    p.add_argument("--results", type=Path, default=Path("results") / "exp4_ablation_epsilon_results.json")
    p.add_argument("--figures-dir", type=Path, default=Path("figures"))
    args = p.parse_args()
    with open(args.results) as f:
        plot_all(json.load(f), args.figures_dir)


if __name__ == "__main__":
    main()
