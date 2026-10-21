"""Figure 11: CC-MMD generative quality against the proxy AUROC (IAF, K-MNIST, epsilon = 0.6).

Reads results/exp4_ablation_proxy_quality_results.json (written by
exp4_ablation_proxy_quality.py, which also calls this script at the end).

Outputs:
  figures/exp4_ablation_proxy_quality_mmd2.pdf   test MMD^2 vs proxy AUROC (Figure 11, left)
  figures/exp4_ablation_proxy_quality_fid.pdf    FID vs proxy AUROC (Figure 11, right)

Usage:
    python experiments/exp4_plot_ablation_proxy_quality.py
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLOR = "#3a7eba"


def aggregate(records: list[dict]):
    """Per label-noise level, the mean proxy AUROC and the mean and std of test MMD^2 and FID,
    ordered by increasing mean AUROC."""
    groups: dict[float, list[dict]] = defaultdict(list)
    for r in records:
        groups[r["label_noise"]].append(r)
    levels = sorted(groups.keys())

    def stat(fn, key):
        return np.array([fn([r[key] for r in groups[ln]]) for ln in levels])

    mean_auroc = stat(np.mean, "auroc")
    order = np.argsort(mean_auroc)
    return (mean_auroc[order], stat(np.mean, "test_mmd2")[order], stat(np.std, "test_mmd2")[order],
            stat(np.mean, "fid")[order], stat(np.std, "fid")[order])


def plot_metric(mean_auroc: np.ndarray, mean_metric: np.ndarray, std_metric: np.ndarray,
                y_label: str, out_path: Path, sci_y: bool = False) -> None:
    """Mean +/- std of a metric against the mean proxy AUROC."""
    fig, ax = plt.subplots(figsize=(13, 8))

    ax.plot(mean_auroc, mean_metric, "-o", color=COLOR, markersize=24, linewidth=8.0, label="CC-MMD")
    ax.legend(fontsize=26)
    ax.fill_between(mean_auroc, mean_metric - std_metric, mean_metric + std_metric, color=COLOR, alpha=0.20)

    ax.set_xlabel("Proxy scores AUROC (↑)", fontsize=36)
    ax.set_ylabel(y_label, fontsize=36)
    ax.tick_params(axis="x", labelsize=33, pad=18)
    ax.tick_params(axis="y", which="both", labelsize=33)
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    if sci_y:
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_fontsize(28)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5, zorder=0)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_all(records: list[dict], figures_dir: Path) -> None:
    """Draw both panels of Figure 11."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    mean_auroc, mean_mmd2, std_mmd2, mean_fid, std_fid = aggregate(records)
    plot_metric(mean_auroc, mean_mmd2, std_mmd2, "Test MMD² (↓)",
                figures_dir / "exp4_ablation_proxy_quality_mmd2.pdf", sci_y=True)
    plot_metric(mean_auroc, mean_fid, std_fid, "FID (↓)",
                figures_dir / "exp4_ablation_proxy_quality_fid.pdf")


def main() -> None:
    p = argparse.ArgumentParser(description="Figure 11 of the paper.")
    p.add_argument("--results", type=Path,
                   default=Path("results") / "exp4_ablation_proxy_quality_results.json")
    p.add_argument("--figures-dir", type=Path, default=Path("figures"))
    args = p.parse_args()
    with open(args.results) as f:
        plot_all(json.load(f), args.figures_dir)


if __name__ == "__main__":
    main()
