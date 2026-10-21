"""Experiment 4, audit-size ablation (Figure 12).

With epsilon = 0.6 (K-MNIST) fixed, varies the audit size m / N in
{5%, 10%, 20%, 30%, 40%, 50%} and trains an IAF with CC-MMD and with audit-only
MMD (naive MMD on the audited MNIST points). To keep the proxy quality fixed
across audit sizes, the fraction eta of flipped audit labels is chosen by
bisection so that the proxy AUROC is 0.7 (within 0.01). Each audit size is
repeated for seeds --seed, ..., --seed + 19.

Every replicate is saved to results/exp4_ablation_audit_size/task_XXXX.json, so
the work can be split across processes with --task-id; a run without --task-id
runs the missing replicates, collects all of them into
results/exp4_ablation_audit_size_results.json and draws Figure 12.

Usage:
    python experiments/exp4_ablation_audit_size.py                # everything, sequentially
    python experiments/exp4_ablation_audit_size.py --task-id 17   # one replicate (0 to 119)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exp4_ablation_epsilon import _make_flow, setup_shared
from exp4_iaf_config import CFG
from exp4_plot_ablation_audit_size import plot_all

from cc_mmd.data import sample_audit_subset
from cc_mmd.data.mnist import build_contaminated_pool, compute_proxy_scores_mnist, dataset_to_tensor, proxy_auroc
from cc_mmd.evaluation.metrics import evaluate_flow
from cc_mmd.losses.mmd import CCMMDLoss, NaiveMMDLoss, median_bandwidth
from cc_mmd.training.train import train_mmd
from cc_mmd.utils.device import get_best_device

AUDIT_FRACTIONS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
N_REPS = 20
TARGET_AUROC = 0.70


def find_label_noise(
    target_auroc: float,
    W_contam: torch.Tensor,
    L_indices: torch.Tensor,
    I_true_L: torch.Tensor,
    I_true: torch.Tensor,
    seed: int,
    tol: float = 0.01,
    max_iter: int = 20,
) -> tuple[float, float, torch.Tensor]:
    """Bisection over the label noise eta so that the proxy AUROC is within tol of target_auroc.

    Returns (eta, achieved AUROC, proxy scores). If even eta = 0 gives an AUROC below
    the target, eta = 0 is used.
    """
    I_tilde_lo = compute_proxy_scores_mnist(W_contam, L_indices, I_true_L, label_noise=0.0, random_state=seed)
    auroc_lo = proxy_auroc(I_tilde_lo, I_true)
    if auroc_lo < target_auroc:
        print(f"  eta = 0 gives AUROC {auroc_lo:.4f} < target {target_auroc:.4f}; using eta = 0")
        return 0.0, auroc_lo, I_tilde_lo

    lo, hi = 0.0, 1.0
    I_tilde_best, eta_best, auroc_best = I_tilde_lo, 0.0, auroc_lo
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        I_tilde_mid = compute_proxy_scores_mnist(W_contam, L_indices, I_true_L, label_noise=mid, random_state=seed)
        auroc_mid = proxy_auroc(I_tilde_mid, I_true)

        if abs(auroc_mid - target_auroc) <= tol:
            return mid, auroc_mid, I_tilde_mid
        if auroc_mid > target_auroc:   # more label noise lowers the AUROC
            lo = mid
        else:
            hi = mid
        if abs(auroc_mid - target_auroc) < abs(auroc_best - target_auroc):
            I_tilde_best, eta_best, auroc_best = I_tilde_mid, mid, auroc_mid

    return eta_best, auroc_best, I_tilde_best


def run_one_rep(
    audit_fraction: float,
    seed: int,
    *,
    mnist_train,
    mnist_test,
    kmnist_train,
    fid_net,
    feat_test: np.ndarray,
    n_steps: int,
    device: torch.device,
) -> dict:
    """Train and evaluate CC-MMD and audit-only MMD for one (audit fraction, seed) pair."""
    torch.manual_seed(seed)
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    W_contam, I_true = build_contaminated_pool(
        mnist_train, kmnist_train, n_total=CFG["n_train"], epsilon=CFG["epsilon"], rng=rng, device=device,
    )
    W_test = dataset_to_tensor(mnist_test, CFG["n_test"], rng=rng, device=device)
    sigma_test = median_bandwidth(W_test)

    N = W_contam.shape[0]
    m = round(audit_fraction * N)
    L_indices = sample_audit_subset(N, m, rng=rng, device=device)
    I_true_L = I_true[L_indices]
    clean_audit_idx = L_indices[I_true_L == 1]
    sigma = (median_bandwidth(W_contam[clean_audit_idx]) if clean_audit_idx.shape[0] >= 2
             else median_bandwidth(W_contam))

    _, auroc, I_tilde = find_label_noise(TARGET_AUROC, W_contam, L_indices, I_true_L, I_true, seed)

    torch.manual_seed(seed)
    flow, _ = train_mmd(_make_flow(device), CCMMDLoss(W_contam, I_tilde, L_indices, I_true_L, sigma),
                        n_steps=n_steps, n_gen=CFG["n_gen"], lr=CFG["lr_mmd"], device=device,
                        desc=f"CC-MMD m={audit_fraction * 100:.0f}% seed={seed}")
    cc = evaluate_flow(flow, W_test, sigma_test, fid_net, feat_test,
                       n_gen_eval=CFG["n_gen_eval"], n_eval_mmd=CFG["n_eval_mmd"], device=device)
    print(f"  m={audit_fraction * 100:.0f}%  seed={seed}  CC-MMD      AUROC={auroc:.4f}  "
          f"test MMD^2={cc['test_mmd2']:.6f}  FID={cc['fid']:.2f}")

    W_audit = W_contam[clean_audit_idx] if clean_audit_idx.shape[0] >= 2 else W_contam
    torch.manual_seed(seed)
    flow_ao, _ = train_mmd(_make_flow(device), NaiveMMDLoss(W_audit, sigma),
                           n_steps=n_steps, n_gen=CFG["n_gen"], lr=CFG["lr_mmd"], device=device,
                           desc=f"audit-only m={audit_fraction * 100:.0f}% seed={seed}")
    ao = evaluate_flow(flow_ao, W_test, sigma_test, fid_net, feat_test,
                       n_gen_eval=CFG["n_gen_eval"], n_eval_mmd=CFG["n_eval_mmd"], device=device)
    print(f"  m={audit_fraction * 100:.0f}%  seed={seed}  audit-only  "
          f"test MMD^2={ao['test_mmd2']:.6f}  FID={ao['fid']:.2f}")

    return {
        "auroc": auroc,
        "test_mmd2": cc["test_mmd2"],
        "fid": cc["fid"],
        "m": m,
        "n_clean_audit": int(clean_audit_idx.shape[0]),
        "audit_only_test_mmd2": ao["test_mmd2"],
        "audit_only_fid": ao["fid"],
    }


def run_task(task_id: int, args, device, shared, task_dir: Path) -> dict:
    """Run replicate task_id (fraction index task_id // n_reps, replicate task_id % n_reps) and save it."""
    frac_idx, rep = divmod(task_id, args.n_reps)
    audit_fraction = AUDIT_FRACTIONS[frac_idx]
    seed = args.seed + rep
    print(f"\n=== task {task_id}: audit size {audit_fraction * 100:.0f}%, replicate {rep}, seed {seed} ===")

    mnist_train, mnist_test, kmnist_train, fid_net, feat_test = shared
    res = run_one_rep(audit_fraction, seed, mnist_train=mnist_train, mnist_test=mnist_test,
                      kmnist_train=kmnist_train, fid_net=fid_net, feat_test=feat_test,
                      n_steps=args.n_steps, device=device)
    record = {"task_id": task_id, "frac_idx": frac_idx, "audit_fraction": audit_fraction,
              "rep": rep, "seed": seed, **res}

    task_dir.mkdir(parents=True, exist_ok=True)
    with open(task_dir / f"task_{task_id:04d}.json", "w") as f:
        json.dump(record, f, indent=2)
    return record


def print_summary(records: list[dict]) -> None:
    print("\nMean over replicates:")
    print(f"  {'m':>4}  {'AUROC':>7}  {'CC-MMD MMD^2':>13}  {'audit-only MMD^2':>17}  {'CC-MMD FID':>11}  "
          f"{'audit-only FID':>15}")
    for af in AUDIT_FRACTIONS:
        rs = [r for r in records if r["audit_fraction"] == af]
        mean = lambda k: np.mean([r[k] for r in rs])
        print(f"  {af * 100:>3.0f}%  {mean('auroc'):>7.4f}  {mean('test_mmd2'):>13.6f}  "
              f"{mean('audit_only_test_mmd2'):>17.6f}  {mean('fid'):>11.2f}  {mean('audit_only_fid'):>15.2f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 4: audit-size ablation (Figure 12)")
    p.add_argument("--device", default="auto",
                   help="'auto' picks the least busy GPU and enables TF32 (as in the paper's runs).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-steps", type=int, default=CFG["n_steps_mmd"], help="Maximum gradient steps per model.")
    p.add_argument("--n-reps", type=int, default=N_REPS, help="Replicates per audit size.")
    p.add_argument("--task-id", type=int, default=None,
                   help=f"Run only this replicate (0 to {len(AUDIT_FRACTIONS)} x n_reps - 1) and exit.")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--figures-dir", default="figures")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_best_device() if args.device == "auto" else torch.device(args.device)
    task_dir = Path(args.results_dir) / "exp4_ablation_audit_size"
    n_tasks = len(AUDIT_FRACTIONS) * args.n_reps

    if args.task_id is not None:
        if not 0 <= args.task_id < n_tasks:
            raise ValueError(f"--task-id must be in [0, {n_tasks - 1}]")
        run_task(args.task_id, args, device, setup_shared(args, device), task_dir)
        return

    done = {}
    for tf in sorted(task_dir.glob("task_*.json")) if task_dir.exists() else []:
        with open(tf) as f:
            rec = json.load(f)
        done[rec["task_id"]] = rec
    print(f"Audit-size ablation: {n_tasks} replicates, {len(done)} already done")

    shared = None
    for task_id in range(n_tasks):
        if task_id not in done:
            shared = shared or setup_shared(args, device)
            done[task_id] = run_task(task_id, args, device, shared, task_dir)

    records = [done[t] for t in range(n_tasks)]
    results_path = Path(args.results_dir) / "exp4_ablation_audit_size_results.json"
    with open(results_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved {results_path}")

    print_summary(records)
    plot_all(records, Path(args.figures_dir))


if __name__ == "__main__":
    main()
