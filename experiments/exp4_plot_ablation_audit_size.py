"""Figure 12: CC-MMD and audit-only generative quality against the audit size
(IAF, K-MNIST, epsilon = 0.6, proxy AUROC about 0.7).

Reads results/exp4_ablation_audit_size_results.json (written by
exp4_ablation_audit_size.py, which also calls this script at the end).

Outputs:
  figures/exp4_ablation_audit_size_mmd2.pdf   test MMD^2 vs audit size (Figure 12, left)
  figures/exp4_ablation_audit_size_fid.pdf    FID vs audit size (Figure 12, right)

Usage:
    python experiments/exp4_plot_ablation_audit_size.py
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

CCMMD_STYLE = dict(color="#3a7eba", label="CC-MMD")
AUDIT_ONLY_STYLE = dict(color="#888888", label="Audit-only", linestyle="--")


def aggregate(records: list[dict]) -> dict:
    """Per audit fraction, the mean and std over replicates of each metric."""
    groups: dict[float, list[dict]] = defaultdict(list)
    for r in records:
        groups[r["audit_fraction"]].append(r)
    fractions = sorted(groups.keys())

    def stats(key: str):
        vals = [[r[key] for r in groups[f]] for f in fractions]
        return np.array([np.mean(v) for v in vals]), np.array([np.std(v) for v in vals])

    return {
        "fractions_pct": np.array(fractions) * 100,
        "cc_mmd": {"mmd2": stats("test_mmd2"), "fid": stats("fid")},
        "audit_only": {"mmd2": stats("audit_only_test_mmd2"), "fid": stats("audit_only_fid")},
    }


def plot_metric(fractions_pct: np.ndarray, cc: tuple, ao: tuple, y_label: str, out_path: Path,
                sci_y: bool = False) -> None:
    """Mean +/- std of CC-MMD and audit-only against the audit size."""
    fig, ax = plt.subplots(figsize=(13, 8))

    cc_mean, cc_std = cc
    ax.plot(fractions_pct, cc_mean, "-o", color=CCMMD_STYLE["color"], markersize=24,
            linewidth=8.0, label=CCMMD_STYLE["label"])
    ax.fill_between(fractions_pct, cc_mean - cc_std, cc_mean + cc_std, color=CCMMD_STYLE["color"], alpha=0.20)

    ao_mean, ao_std = ao
    ax.plot(fractions_pct, ao_mean, marker="s", color=AUDIT_ONLY_STYLE["color"],
            linestyle=AUDIT_ONLY_STYLE["linestyle"], markersize=18, linewidth=6.0,
            label=AUDIT_ONLY_STYLE["label"])
    ax.fill_between(fractions_pct, ao_mean - ao_std, ao_mean + ao_std,
                    color=AUDIT_ONLY_STYLE["color"], alpha=0.15)

    ax.set_xlabel("Audit size $m$ (% of training pool)", fontsize=36)
    ax.set_ylabel(y_label, fontsize=36)
    ax.tick_params(axis="x", labelsize=33, pad=18)
    ax.tick_params(axis="y", which="both", labelsize=33)
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    if sci_y:
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_fontsize(28)
    ax.legend(fontsize=26)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5, zorder=0)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_all(records: list[dict], figures_dir: Path) -> None:
    """Draw both panels of Figure 12."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    data = aggregate(records)
    x = data["fractions_pct"]
    plot_metric(x, data["cc_mmd"]["mmd2"], data["audit_only"]["mmd2"], "Test MMD² (↓)",
                figures_dir / "exp4_ablation_audit_size_mmd2.pdf", sci_y=True)
    plot_metric(x, data["cc_mmd"]["fid"], data["audit_only"]["fid"], "FID (↓)",
                figures_dir / "exp4_ablation_audit_size_fid.pdf")


def main() -> None:
    p = argparse.ArgumentParser(description="Figure 12 of the paper.")
    p.add_argument("--results", type=Path, default=Path("results") / "exp4_ablation_audit_size_results.json")
    p.add_argument("--figures-dir", type=Path, default=Path("figures"))
    args = p.parse_args()
    with open(args.results) as f:
        plot_all(json.load(f), args.figures_dir)


if __name__ == "__main__":
    main()
