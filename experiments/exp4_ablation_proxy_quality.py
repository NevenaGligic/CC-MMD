"""Experiment 4, proxy-quality ablation (Figure 11).

With epsilon = 0.6 (K-MNIST) and a 10% audit fixed, varies the proxy quality by
flipping a fraction eta in {0.0, 0.1, ..., 0.5} of the audit labels before fitting
the proxy, and trains an IAF with CC-MMD. Each level is repeated for seeds
--seed, ..., --seed + 19, and the results are reported against the achieved proxy AUROC.

Every replicate is saved to results/exp4_ablation_proxy_quality/task_XXXX.json, so
the work can be split across processes with --task-id; a run without --task-id
runs the missing replicates, collects all of them into
results/exp4_ablation_proxy_quality_results.json and draws Figure 11.

Usage:
    python experiments/exp4_ablation_proxy_quality.py                # everything, sequentially
    python experiments/exp4_ablation_proxy_quality.py --task-id 17   # one replicate (0 to 119)
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
from exp4_plot_ablation_proxy_quality import plot_all

from cc_mmd.data import sample_audit_subset
from cc_mmd.data.mnist import build_contaminated_pool, compute_proxy_scores_mnist, dataset_to_tensor, proxy_auroc
from cc_mmd.evaluation.metrics import evaluate_flow
from cc_mmd.losses.mmd import CCMMDLoss, median_bandwidth
from cc_mmd.training.train import train_mmd
from cc_mmd.utils.device import get_best_device

NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
N_REPS = 20


def run_one_rep(
    label_noise: float,
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
    """Train and evaluate CC-MMD for one (label noise, seed) pair; returns auroc, test_mmd2 and fid."""
    torch.manual_seed(seed)
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    W_contam, I_true = build_contaminated_pool(
        mnist_train, kmnist_train, n_total=CFG["n_train"], epsilon=CFG["epsilon"], rng=rng, device=device,
    )
    W_test = dataset_to_tensor(mnist_test, CFG["n_test"], rng=rng, device=device)
    sigma_test = median_bandwidth(W_test)

    N = W_contam.shape[0]
    m = round(CFG["audit_fraction"] * N)
    L_indices = sample_audit_subset(N, m, rng=rng, device=device)
    I_true_L = I_true[L_indices]
    clean_audit_idx = L_indices[I_true_L == 1]
    sigma = (median_bandwidth(W_contam[clean_audit_idx]) if clean_audit_idx.shape[0] >= 2
             else median_bandwidth(W_contam))

    I_tilde = compute_proxy_scores_mnist(W_contam, L_indices, I_true_L, label_noise=label_noise, random_state=seed)
    auroc = proxy_auroc(I_tilde, I_true)

    torch.manual_seed(seed)
    flow, _ = train_mmd(_make_flow(device), CCMMDLoss(W_contam, I_tilde, L_indices, I_true_L, sigma),
                        n_steps=n_steps, n_gen=CFG["n_gen"], lr=CFG["lr_mmd"], device=device,
                        desc=f"CC-MMD eta={label_noise:.1f} seed={seed}")
    metrics = evaluate_flow(flow, W_test, sigma_test, fid_net, feat_test,
                            n_gen_eval=CFG["n_gen_eval"], n_eval_mmd=CFG["n_eval_mmd"], device=device)
    print(f"  eta={label_noise:.1f}  seed={seed}  AUROC={auroc:.4f}  "
          f"test MMD^2={metrics['test_mmd2']:.6f}  FID={metrics['fid']:.2f}")
    return {"auroc": auroc, "test_mmd2": metrics["test_mmd2"], "fid": metrics["fid"]}


def run_task(task_id: int, args, device, shared, task_dir: Path) -> dict:
    """Run replicate task_id (noise index task_id // n_reps, replicate task_id % n_reps) and save it."""
    noise_idx, rep = divmod(task_id, args.n_reps)
    label_noise = NOISE_LEVELS[noise_idx]
    seed = args.seed + rep
    print(f"\n=== task {task_id}: eta = {label_noise:.1f}, replicate {rep}, seed {seed} ===")

    mnist_train, mnist_test, kmnist_train, fid_net, feat_test = shared
    res = run_one_rep(label_noise, seed, mnist_train=mnist_train, mnist_test=mnist_test,
                      kmnist_train=kmnist_train, fid_net=fid_net, feat_test=feat_test,
                      n_steps=args.n_steps, device=device)
    record = {"task_id": task_id, "noise_idx": noise_idx, "label_noise": label_noise,
              "rep": rep, "seed": seed, **res}

    task_dir.mkdir(parents=True, exist_ok=True)
    with open(task_dir / f"task_{task_id:04d}.json", "w") as f:
        json.dump(record, f, indent=2)
    return record


def print_summary(records: list[dict]) -> None:
    print("\nMean over replicates (test MMD^2 and FID as mean +/- std):")
    print(f"  {'eta':>4}  {'AUROC':>7}  {'test MMD^2':>22}  {'FID':>16}")
    for ln in NOISE_LEVELS:
        rs = [r for r in records if r["label_noise"] == ln]
        mmd = np.array([r["test_mmd2"] for r in rs])
        fid = np.array([r["fid"] for r in rs])
        print(f"  {ln:>4.1f}  {np.mean([r['auroc'] for r in rs]):>7.4f}  "
              f"{mmd.mean():>10.6f} +- {mmd.std():<9.6f}  {fid.mean():>7.2f} +- {fid.std():<6.2f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 4: proxy-quality ablation (Figure 11)")
    p.add_argument("--device", default="auto",
                   help="'auto' picks the least busy GPU and enables TF32 (as in the paper's runs).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-steps", type=int, default=CFG["n_steps_mmd"], help="Maximum gradient steps per model.")
    p.add_argument("--n-reps", type=int, default=N_REPS, help="Replicates per label-noise level.")
    p.add_argument("--task-id", type=int, default=None,
                   help=f"Run only this replicate (0 to {len(NOISE_LEVELS)} x n_reps - 1) and exit.")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--figures-dir", default="figures")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_best_device() if args.device == "auto" else torch.device(args.device)
    task_dir = Path(args.results_dir) / "exp4_ablation_proxy_quality"
    n_tasks = len(NOISE_LEVELS) * args.n_reps

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
    print(f"Proxy-quality ablation: {n_tasks} replicates, {len(done)} already done")

    shared = None
    for task_id in range(n_tasks):
        if task_id not in done:
            shared = shared or setup_shared(args, device)
            done[task_id] = run_task(task_id, args, device, shared, task_dir)

    records = [done[t] for t in range(n_tasks)]
    results_path = Path(args.results_dir) / "exp4_ablation_proxy_quality_results.json"
    with open(results_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved {results_path}")

    print_summary(records)
    plot_all(records, Path(args.figures_dir))


if __name__ == "__main__":
    main()
