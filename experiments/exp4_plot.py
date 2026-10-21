"""Figure 3: test MMD^2 and FID of each architecture and loss, mean +/- std over seeds.

Reads results/exp4_results.json (written by exp4_collect_results.py).

Outputs:
  figures/exp4_mmd2_kmnist.pdf     figures/exp4_mmd2_omniglot.pdf
  figures/exp4_fid_kmnist.pdf      figures/exp4_fid_omniglot.pdf

Usage:
    python experiments/exp4_plot.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.ticker import LogLocator
from matplotlib.transforms import blended_transform_factory

# Bars of each architecture group: (label, model, colour).
ARCH_GROUPS = [
    ("iaf", "IAF", [
        ("SWD",    "swd_contam",   "#1565c0"),
        ("MMD",    "mmd_contam",   "#5c85d4"),
        ("CC-MMD", "ccmmd_contam", "#9dc3e6"),
    ]),
    ("vae", "VAE", [
        ("MSE",    "vae_mse_contam",   "#1b5e20"),
        ("Huber",  "vae_huber_contam", "#388e3c"),
        ("MMD",    "vae_mmd_contam",   "#66bb6a"),
        ("CC-MMD", "vae_ccmmd_contam", "#a5d6a7"),
    ]),
    ("gmmn", "GMMN", [
        ("GAN",    "gmmn_gan_contam",   "#4a148c"),
        ("MMD",    "gmmn_mmd_contam",   "#7b1fa2"),
        ("CC-MMD", "gmmn_ccmmd_contam", "#ce93d8"),
    ]),
]

BAR_WIDTH = 0.9
GROUP_GAP = 0.2


def compute_stats(data: dict, arch: str, contam: str, model: str, metric: str) -> tuple[float, float, int]:
    """Mean, sample std and number of seeds of a metric."""
    values = [s[model][metric] for s in data.get(arch, {}).get(contam, {}).values()
              if model in s and metric in s[model]]
    if not values:
        return float("nan"), float("nan"), 0
    arr = np.array(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1) if len(arr) > 1 else 0.0), len(arr)


def _log_top(val_max: float, bottom: float, fraction: float = 0.90) -> float:
    """Top of a log axis that puts val_max at `fraction` of the axis height."""
    return bottom * np.exp(np.log(val_max / bottom) / fraction)


def _log_label(x: float, _) -> str:
    """Tick label 10^n or m x 10^n."""
    exp = int(np.floor(np.log10(x) + 1e-9))
    mantissa = round(x / 10 ** exp)
    return f"$10^{{{exp}}}$" if mantissa == 1 else f"${mantissa}\\times10^{{{exp}}}$"


def build_positions() -> tuple[list[float], list[float], list[float]]:
    """x positions of the bars, the group centres and the separators between groups."""
    bar_x, group_center_x, sep_x = [], [], []
    cursor = 0.0
    for _, _, bars in ARCH_GROUPS:
        n = len(bars)
        bar_x.extend(cursor + i * BAR_WIDTH for i in range(n))
        group_center_x.append(cursor + (n - 1) * BAR_WIDTH / 2)
        cursor += n * BAR_WIDTH + GROUP_GAP
        sep_x.append(cursor - GROUP_GAP / 2)
    return bar_x, group_center_x, sep_x[:-1]


def plot_metric(metric: str, ylabel: str, data: dict, contam: str, output_path: Path,
                log_subs: list[int]) -> None:
    """Bar chart (log scale) of one metric for every architecture and loss."""
    bar_x, group_center_x, sep_x = build_positions()
    fig, ax = plt.subplots(figsize=(max(14, len(bar_x) * 1.5), 7))

    stats = [compute_stats(data, arch, contam, model, metric)
             for arch, _, bars in ARCH_GROUPS for _, model, _ in bars]
    tops = [m + s for m, s, _ in stats if not np.isnan(m)]
    if not tops:
        print(f"  No data for {metric} ({contam}); skipping {output_path.name}")
        return
    val_max = max(tops)
    val_fmt = "{:.3f}" if val_max < 10 else "{:.1f}"

    colors = [c for _, _, bars in ARCH_GROUPS for _, _, c in bars]
    for x, (mu, sigma, n), color in zip(bar_x, stats, colors):
        if np.isnan(mu):
            continue
        ax.bar(x, mu, width=BAR_WIDTH * 0.88, color=color, edgecolor="black", linewidth=0.6, zorder=3)
        if n > 1:
            ax.errorbar(x, mu, yerr=sigma, fmt="none", ecolor="black",
                        elinewidth=2.5, capsize=7, capthick=2.5, zorder=5)
        ax.text(x, mu + (sigma if n > 1 else 0), val_fmt.format(mu),
                ha="center", va="bottom", fontsize=24, color="#222222", zorder=4)

    labels = [lbl for _, _, bars in ARCH_GROUPS for lbl, _, _ in bars]
    ax.set_xticks(bar_x)
    ax.set_xticklabels(labels, fontsize=22, ha="center")
    for tick, lbl in zip(ax.get_xticklabels(), labels):
        if lbl == "CC-MMD":
            tick.set_fontweight("bold")
    ax.tick_params(axis="x", pad=8)
    ax.tick_params(axis="y", which="both", labelsize=28)

    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for (_, group_label, _), cx in zip(ARCH_GROUPS, group_center_x):
        ax.text(cx, -0.09, group_label, transform=trans, ha="center", va="top",
                fontsize=32, fontweight="bold", color="#333333", clip_on=False)
    for sx in sep_x:
        ax.axvline(sx, color="gray", linewidth=0.7, linestyle="--", alpha=0.45, zorder=1)

    ax.set_ylabel(ylabel, fontsize=32)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5, zorder=0)
    ax.set_xlim(bar_x[0] - BAR_WIDTH * 0.7, bar_x[-1] + BAR_WIDTH * 0.7)

    ax.set_yscale("log")
    bottom = min([m for m, _, _ in stats if not np.isnan(m) and m > 0] or [1e-3]) * 0.5
    ax.set_ylim(bottom=bottom, top=_log_top(val_max, bottom))
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=log_subs, numticks=12))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_log_label))
    ax.yaxis.set_minor_formatter(plt.NullFormatter())

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=400)
    plt.close(fig)
    print(f"Saved {output_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Figure 3 of the paper.")
    p.add_argument("--results", type=Path, default=Path("results") / "exp4_results.json")
    p.add_argument("--figures-dir", type=Path, default=Path("figures"))
    args = p.parse_args()

    with open(args.results) as f:
        data = json.load(f)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    for contam in ["kmnist", "omniglot"]:
        plot_metric("test_mmd2", "Test MMD² ↓", data, contam,
                    args.figures_dir / f"exp4_mmd2_{contam}.pdf", log_subs=[1])
        plot_metric("fid", "FID ↓", data, contam,
                    args.figures_dir / f"exp4_fid_{contam}.pdf", log_subs=[1, 2, 5])


if __name__ == "__main__":
    main()
