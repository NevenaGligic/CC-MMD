"""Experiment 1: MMD estimation under increasing contamination (Figure 1, Appendix).

Compares the oracle, naive, audit-only and CC-MMD estimators of MMD^2(p, q) = 0
on a ring of Gaussians contaminated by a gradient background, for
epsilon in {0.0, 0.1, ..., 0.6} with 50 replicates each.

Outputs:
  figures/sim_visualization.pdf   data setup at epsilon = 0.3 (Appendix)
  figures/contamination_rate.pdf  mean +/- std of the four estimators vs epsilon (Figure 1)
  results/exp1_results.pt         all estimates and proxy AUROCs

Usage:
    python experiments/exp1.py
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cc_mmd.data import sample_audit_subset
from cc_mmd.data.synthetic import (
    compute_proxy_scores_exp1,
    make_contaminated_pool,
    sample_ring_gaussians,
)
from cc_mmd.estimator import audit_only_mmd2, cc_mmd2, naive_mmd2, oracle_mmd2
from cc_mmd.kernels import gaussian_kernel

CFG = dict(
    n=400,            # reference sample size
    N=400,            # pooled sample size
    m=80,             # audit size
    sigma_k=1.5,      # kernel bandwidth
    R=5.0,            # ring radius
    sigma_q=0.6,      # cluster std
    K=5,              # number of clusters
    sigma_cls=1.0,    # proxy classifier noise std
    epsilons=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
)

ESTIMATORS = ["oracle", "naive", "audit_only", "cc_mmd"]


def run_one_rep(
    epsilon: float,
    cfg: dict,
    *,
    rng: torch.Generator,
    device: torch.device,
) -> dict:
    """One replicate: the four MMD^2 estimates and the proxy AUROC (NaN when epsilon = 0)."""
    kernel_fn = lambda A, B: gaussian_kernel(A, B, cfg["sigma_k"])

    X = sample_ring_gaussians(cfg["n"], R=cfg["R"], sigma_q=cfg["sigma_q"],
                              K=cfg["K"], rng=rng, device=device)
    W, I = make_contaminated_pool(cfg["N"], epsilon, R=cfg["R"],
                                  sigma_q=cfg["sigma_q"], K=cfg["K"],
                                  rng=rng, device=device)
    I_tilde = compute_proxy_scores_exp1(W, R=cfg["R"], sigma_cls=cfg["sigma_cls"], rng=rng)
    L_indices = sample_audit_subset(cfg["N"], cfg["m"], rng=rng, device=device)
    I_true_L = I[L_indices]

    results = {}
    with torch.no_grad():
        results["oracle"] = oracle_mmd2(X, W, I, kernel_fn).item()
        results["naive"] = naive_mmd2(X, W, kernel_fn).item()
        results["audit_only"] = audit_only_mmd2(X, W, L_indices, I_true_L, kernel_fn).item()
        results["cc_mmd"] = cc_mmd2(X, W, I_tilde, L_indices, I_true_L, kernel_fn).item()

    I_np = I.cpu().numpy()
    has_both = 0 < I_np.sum() < len(I_np)
    results["proxy_auroc"] = roc_auc_score(I_np, I_tilde.cpu().numpy()) if has_both else float("nan")
    return results


def plot_sim_visualization(
    cfg: dict,
    *,
    epsilon_vis: float = 0.3,
    seed: int = 42,
    output_path: Path,
    device: torch.device,
) -> None:
    """Three panels: clean reference and target samples, the contaminated pool, and the proxy scores."""
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    X = sample_ring_gaussians(cfg["n"], R=cfg["R"], sigma_q=cfg["sigma_q"],
                              K=cfg["K"], rng=rng, device=device)
    W, I = make_contaminated_pool(cfg["N"], epsilon_vis, R=cfg["R"],
                                  sigma_q=cfg["sigma_q"], K=cfg["K"],
                                  rng=rng, device=device)
    I_tilde = compute_proxy_scores_exp1(W, R=cfg["R"], sigma_cls=cfg["sigma_cls"], rng=rng)

    X_np = X.cpu().numpy()
    W_np = W.cpu().numpy()
    I_np = I.cpu().numpy()
    I_tilde_np = I_tilde.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    ax = axes[0]
    ax.scatter(X_np[:, 0], X_np[:, 1], s=8, alpha=0.5, color="#3a86ff", label="Reference $p$")
    rng2 = torch.Generator(device=device)
    rng2.manual_seed(seed + 1000)
    Y_clean = sample_ring_gaussians(cfg["N"], R=cfg["R"], sigma_q=cfg["sigma_q"],
                                    K=cfg["K"], rng=rng2, device=device).cpu().numpy()
    ax.scatter(Y_clean[:, 0], Y_clean[:, 1], s=8, alpha=0.5, color="#ff6b6b", label="Target $q$")
    ax.set_title("(a) Clean data")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend(markerscale=2, fontsize=8)
    ax.set_aspect("equal")

    ax = axes[1]
    inliers = W_np[I_np == 1]
    contams = W_np[I_np == 0]
    ax.scatter(contams[:, 0], contams[:, 1], s=8, alpha=0.5,
               color="#f4a261", label=f"Contaminants ($z$, {len(contams)})")
    ax.scatter(inliers[:, 0], inliers[:, 1], s=8, alpha=0.6,
               color="#2a9d8f", label=f"Inliers ($q$, {len(inliers)})")
    ax.set_title(f"(b) Contaminated pool ($\\epsilon={epsilon_vis}$)")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend(markerscale=2, fontsize=8)
    ax.set_aspect("equal")

    ax = axes[2]
    sc = ax.scatter(W_np[:, 0], W_np[:, 1], c=I_tilde_np, cmap="RdYlGn",
                    s=10, alpha=0.8, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, label="Proxy score $\\tilde{I}_j$")
    ax.set_title("(c) Proxy membership scores")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def run_contamination_sweep(
    cfg: dict,
    n_reps: int,
    base_seed: int,
    device: torch.device,
) -> dict:
    """Run all replicates; returns {name: array of shape (n_epsilons, n_reps)} for each
    estimator and for "proxy_auroc". Replicate r at epsilon index i uses seed
    base_seed + 10000 * i + r."""
    epsilons = cfg["epsilons"]
    names = ESTIMATORS + ["proxy_auroc"]
    results = {name: np.full((len(epsilons), n_reps), np.nan) for name in names}

    for i, eps in enumerate(epsilons):
        for rep in tqdm(range(n_reps), desc=f"eps={eps:.1f}", leave=False):
            rng = torch.Generator(device=device)
            rng.manual_seed(base_seed + i * 10_000 + rep)
            rep_results = run_one_rep(eps, cfg, rng=rng, device=device)
            for name in names:
                results[name][i, rep] = rep_results[name]

    return results


def print_proxy_auroc(results: dict, epsilons: list) -> None:
    """Print the proxy AUROC (mean +/- sample std over replicates) per epsilon and over all epsilon > 0."""
    auroc = results["proxy_auroc"]
    print("\nProxy AUROC against the true membership indicators (mean +/- std over replicates):")
    for i, eps in enumerate(epsilons):
        if eps > 0:
            print(f"  epsilon = {eps:.1f}: {np.mean(auroc[i]):.3f} +/- {np.std(auroc[i], ddof=1):.3f}")
    pooled = auroc[[i for i, eps in enumerate(epsilons) if eps > 0]].ravel()
    print(f"  all epsilon > 0: {np.mean(pooled):.3f} +/- {np.std(pooled, ddof=1):.3f}")


def plot_contamination_rate(results: dict, epsilons: list, output_path: Path) -> None:
    """Mean +/- std of each estimator against the contamination rate (Figure 1)."""
    styles = {
        "oracle":     dict(color="black",   linestyle="-",  label="Oracle"),
        "cc_mmd":     dict(color="#7b1fa2", linestyle="-",  label="CC-MMD"),
        "naive":      dict(color="#d62728", linestyle="-",  label="Naïve MMD"),
        "audit_only": dict(color="#17becf", linestyle="--", label="Audit-only"),
    }
    plot_order = ["audit_only", "naive", "cc_mmd", "oracle"]

    fig, ax = plt.subplots(figsize=(13, 8))
    eps_arr = np.array(epsilons)

    ax.axhline(0.0, color="black", linewidth=1.5, linestyle="--",
               label="True $\\mathrm{MMD}^2=0$", zorder=3)

    for name in plot_order:
        sty = styles[name]
        arr = results[name]
        mean = np.nanmean(arr, axis=1)
        std = np.nanstd(arr, axis=1)
        ax.fill_between(eps_arr, mean - std, mean + std,
                        color=sty["color"], alpha=0.25, zorder=1)
        ax.plot(eps_arr, mean,
                color=sty["color"], linestyle=sty["linestyle"], marker="o",
                label=sty["label"], markersize=16, linewidth=5.0, zorder=2)

    handles, labels = ax.get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))
    legend_order = ["True $\\mathrm{MMD}^2=0$"] + [styles[k]["label"] for k in ["oracle", "audit_only", "naive", "cc_mmd"]]
    ordered_handles = [label_to_handle[lbl] for lbl in legend_order]

    ax.set_xlabel(r"Contamination rate $\epsilon$", fontsize=36)
    ax.set_ylabel(r"Test MMD$^2$ ↓", fontsize=36, labelpad=12)
    ax.set_xticks(epsilons)
    ax.set_xticklabels([f"{e:.1f}" for e in epsilons])
    ax.tick_params(axis="x", labelsize=33, pad=18)
    ax.tick_params(axis="y", labelsize=33)
    ax.legend(ordered_handles, legend_order, fontsize=26)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5, zorder=0)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22, left=0.18)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 1: MMD estimation under contamination")
    parser.add_argument("--n-reps", type=int, default=50, help="Replicates per contamination level.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--figures-dir", type=str, default="figures")
    parser.add_argument("--results-dir", type=str, default="results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.figures_dir)
    results_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}  |  Replicates: {args.n_reps}  |  Seed: {args.seed}")

    print("\n[1/3] Data setup figure ...")
    plot_sim_visualization(CFG, epsilon_vis=0.3, seed=args.seed,
                           output_path=output_dir / "sim_visualization.pdf", device=device)

    print(f"\n[2/3] Contamination sweep ({args.n_reps} replicates x {len(CFG['epsilons'])} epsilon values) ...")
    results = run_contamination_sweep(CFG, n_reps=args.n_reps, base_seed=args.seed, device=device)
    results_path = results_dir / "exp1_results.pt"
    torch.save({"results": results, "epsilons": CFG["epsilons"], "cfg": CFG}, results_path)
    print(f"Results saved to {results_path}")
    print_proxy_auroc(results, CFG["epsilons"])

    print("\n[3/3] Figure 1 ...")
    plot_contamination_rate(results, CFG["epsilons"], output_dir / "contamination_rate.pdf")

    print("\nExperiment 1 complete.")


if __name__ == "__main__":
    main()
