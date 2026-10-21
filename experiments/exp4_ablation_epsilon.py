"""Experiment 4, contamination-rate ablation (Figure 4).

For each epsilon in {0.0, 0.1, ..., 0.8} and each of 20 replicates, builds an
MNIST pool contaminated with K-MNIST, fits the proxy with label noise eta = 0.3
(proxy AUROC about 0.7), and trains an IAF with five losses:

  oracle      naive MMD on the genuine MNIST points of the pool
  audit_only  naive MMD on the audited MNIST points
  naive_mmd   naive MMD on the whole pool
  swd         SWD on the whole pool
  cc_mmd      CC-MMD

Replicate r at epsilon index i uses seed --seed + 10000 i + r.

Every replicate is saved to results/exp4_ablation_epsilon/task_XXXX.json, so the
work can be split across processes with --task-id (one (epsilon, replicate) pair
per task); a run without --task-id runs the missing replicates, collects all of
them into results/exp4_ablation_epsilon_results.json and draws Figure 4.

Usage:
    python experiments/exp4_ablation_epsilon.py                 # everything, sequentially
    python experiments/exp4_ablation_epsilon.py --task-id 17    # one replicate (0 to 179)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torchvision.transforms as T
from torchvision.datasets import MNIST as RawMNIST

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exp4_iaf_config import CFG
from exp4_plot_ablation_epsilon import MODEL_ORDER, plot_all

from cc_mmd.data import sample_audit_subset
from cc_mmd.data.mnist import (
    build_contaminated_pool,
    compute_proxy_scores_mnist,
    dataset_to_tensor,
    load_datasets,
    proxy_auroc,
)
from cc_mmd.evaluation.fid import extract_features, load_fid_net, train_fid_net
from cc_mmd.evaluation.metrics import evaluate_flow
from cc_mmd.losses.mmd import CCMMDLoss, NaiveMMDLoss, median_bandwidth
from cc_mmd.losses.swd import SWDLoss
from cc_mmd.models.flow import make_iaf
from cc_mmd.training.train import train_mmd
from cc_mmd.utils.device import get_best_device

EPSILONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
N_REPS = 20


def _make_flow(device: torch.device) -> torch.nn.Module:
    return make_iaf(
        hidden_features=CFG["hidden_features"],
        num_layers=CFG["num_layers"],
        num_blocks_per_layer=CFG["num_blocks_per_layer"],
        dropout_prob=CFG["dropout_prob"],
        use_residual_blocks=CFG["use_residual_blocks"],
    ).to(device)


def run_one_rep(
    epsilon: float,
    seed: int,
    *,
    mnist_train,
    mnist_test,
    kmnist_train,
    fid_net,
    feat_test: np.ndarray,
    n_steps: int,
    label_noise: float,
    device: torch.device,
) -> tuple[Dict[str, dict], Optional[float]]:
    """Train and evaluate the five models for one (epsilon, seed) pair.

    Returns ({model: {"test_mmd2", "fid"}}, proxy AUROC or None when the audit has one class).
    """
    torch.manual_seed(seed)
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    # ---- Data ---------------------------------------------------------------
    if epsilon == 0.0:
        W_contam = dataset_to_tensor(mnist_train, CFG["n_train"], rng=rng, device=device)
        I_true = torch.ones(CFG["n_train"], dtype=torch.int32, device=device)
    else:
        W_contam, I_true = build_contaminated_pool(
            mnist_train, kmnist_train, n_total=CFG["n_train"], epsilon=epsilon, rng=rng, device=device,
        )
    W_test = dataset_to_tensor(mnist_test, CFG["n_test"], rng=rng, device=device)
    sigma_test = median_bandwidth(W_test)

    # ---- Audit, bandwidth and proxy -----------------------------------------
    N = W_contam.shape[0]
    m = round(CFG["audit_fraction"] * N)
    L_indices = sample_audit_subset(N, m, rng=rng, device=device)
    I_true_L = I_true[L_indices]
    clean_audit_idx = L_indices[I_true_L == 1]
    n_clean_audit = clean_audit_idx.shape[0]

    sigma = median_bandwidth(W_contam[clean_audit_idx]) if n_clean_audit >= 2 else median_bandwidth(W_contam)

    if (I_true_L == 1).any() and (I_true_L == 0).any():
        I_tilde = compute_proxy_scores_mnist(W_contam, L_indices, I_true_L,
                                             label_noise=label_noise, random_state=seed)
        auroc = proxy_auroc(I_tilde, I_true)
    else:
        I_tilde = torch.ones(N, device=device)
        auroc = None

    # ---- Losses -------------------------------------------------------------
    W_clean = W_contam[I_true == 1]
    if W_clean.shape[0] < 2:
        W_clean = W_contam
    W_audit = W_contam[clean_audit_idx] if n_clean_audit >= 2 else W_clean

    losses = {
        "oracle":     NaiveMMDLoss(W_clean, sigma),
        "audit_only": NaiveMMDLoss(W_audit, sigma),
        "naive_mmd":  NaiveMMDLoss(W_contam, sigma),
        "swd":        SWDLoss(W_contam, n_projections=CFG["swd_projections"]),
        "cc_mmd":     CCMMDLoss(W_contam, I_tilde, L_indices, I_true_L, sigma),
    }

    # ---- Train and evaluate -------------------------------------------------
    results: Dict[str, dict] = {}
    for name in MODEL_ORDER:
        torch.manual_seed(seed)
        flow, _ = train_mmd(_make_flow(device), losses[name], n_steps=n_steps, n_gen=CFG["n_gen"],
                            lr=CFG["lr_mmd"], device=device, desc=f"{name} eps={epsilon:.1f}")
        metrics = evaluate_flow(flow, W_test, sigma_test, fid_net, feat_test,
                                n_gen_eval=CFG["n_gen_eval"], n_eval_mmd=CFG["n_eval_mmd"], device=device)
        results[name] = {"test_mmd2": metrics["test_mmd2"], "fid": metrics["fid"]}
        print(f"  eps={epsilon:.1f}  {name:<11}  test MMD^2 = {metrics['test_mmd2']:.5f}  FID = {metrics['fid']:.2f}")

    return results, auroc


def setup_shared(args: argparse.Namespace, device: torch.device):
    """Load the datasets and the FID feature extractor (training it if no saved copy exists)."""
    data_dir = Path(args.data_dir)
    mnist_train, mnist_test, kmnist_train = load_datasets(data_dir)

    fid_ckpt = Path(args.checkpoint_dir) / "fid_net.pt"
    if fid_ckpt.exists():
        print(f"Loading FID feature extractor from {fid_ckpt}")
        fid_net = load_fid_net(fid_ckpt, device=device)
    else:
        print("Training FID feature extractor on clean MNIST ...")
        raw_train = RawMNIST(data_dir, train=True, transform=T.ToTensor(), download=False)
        raw_images = torch.stack([raw_train[i][0] for i in range(len(raw_train))])
        raw_labels = torch.tensor([raw_train[i][1] for i in range(len(raw_train))])
        fid_net = train_fid_net(raw_images, raw_labels, device=device,
                                n_epochs=CFG["fid_epochs"], save_path=fid_ckpt)

    raw_test = RawMNIST(data_dir, train=False, transform=T.ToTensor(), download=False)
    test_images = torch.stack([raw_test[i][0] for i in range(CFG["n_test"])])
    return mnist_train, mnist_test, kmnist_train, fid_net, extract_features(fid_net, test_images)


def run_task(task_id: int, args, device, shared, task_dir: Path) -> dict:
    """Run replicate task_id (epsilon index task_id // n_reps, replicate task_id % n_reps) and save it."""
    eps_idx, rep = divmod(task_id, args.n_reps)
    eps = EPSILONS[eps_idx]
    seed = args.seed + eps_idx * 10_000 + rep
    print(f"\n=== task {task_id}: epsilon = {eps:.1f}, replicate {rep}, seed {seed} ===")

    mnist_train, mnist_test, kmnist_train, fid_net, feat_test = shared
    results, auroc = run_one_rep(eps, seed, mnist_train=mnist_train, mnist_test=mnist_test,
                                 kmnist_train=kmnist_train, fid_net=fid_net, feat_test=feat_test,
                                 n_steps=args.n_steps, label_noise=args.label_noise, device=device)
    record = {"task_id": task_id, "epsilon_idx": eps_idx, "epsilon": eps, "rep": rep, "seed": seed,
              "proxy_auroc": auroc}
    for name in MODEL_ORDER:
        record[f"{name}_fid"] = results[name]["fid"]
        record[f"{name}_mmd2"] = results[name]["test_mmd2"]

    task_dir.mkdir(parents=True, exist_ok=True)
    with open(task_dir / f"task_{task_id:04d}.json", "w") as f:
        json.dump(record, f, indent=2)
    return record


def print_summary(records: list[dict]) -> None:
    aurocs = [r["proxy_auroc"] for r in records if r["epsilon"] > 0 and r["proxy_auroc"] is not None]
    print(f"\nProxy AUROC over all replicates with epsilon > 0: "
          f"{np.mean(aurocs):.3f} +/- {np.std(aurocs, ddof=1):.3f}")
    for metric, label in [("mmd2", "test MMD^2"), ("fid", "FID")]:
        print(f"\nMean {label} over replicates:")
        print(f"  {'eps':>4}" + "".join(f"  {n:>11}" for n in MODEL_ORDER))
        for eps in EPSILONS:
            rs = [r for r in records if r["epsilon"] == eps]
            fmt = "{:>11.6f}" if metric == "mmd2" else "{:>11.2f}"
            print(f"  {eps:>4.1f}" + "".join("  " + fmt.format(np.mean([r[f'{n}_{metric}'] for r in rs]))
                                             for n in MODEL_ORDER))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 4: contamination-rate ablation (Figure 4)")
    p.add_argument("--device", default="auto",
                   help="'auto' picks the least busy GPU and enables TF32 (as in the paper's runs).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-steps", type=int, default=CFG["n_steps_mmd"], help="Maximum gradient steps per model.")
    p.add_argument("--n-reps", type=int, default=N_REPS, help="Replicates per contamination rate.")
    p.add_argument("--label-noise", type=float, default=0.3,
                   help="eta: fraction of audit labels flipped before fitting the proxy.")
    p.add_argument("--task-id", type=int, default=None,
                   help=f"Run only this replicate (0 to {len(EPSILONS)} x n_reps - 1) and exit.")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--figures-dir", default="figures")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_best_device() if args.device == "auto" else torch.device(args.device)
    task_dir = Path(args.results_dir) / "exp4_ablation_epsilon"
    n_tasks = len(EPSILONS) * args.n_reps

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
    print(f"Contamination-rate ablation: {n_tasks} replicates, {len(done)} already done")

    shared = None
    for task_id in range(n_tasks):
        if task_id not in done:
            shared = shared or setup_shared(args, device)
            done[task_id] = run_task(task_id, args, device, shared, task_dir)

    records = [done[t] for t in range(n_tasks)]
    results_path = Path(args.results_dir) / "exp4_ablation_epsilon_results.json"
    with open(results_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved {results_path}")

    print_summary(records)
    plot_all(records, Path(args.figures_dir))


if __name__ == "__main__":
    main()
