"""Experiment 4, IAF: generative modelling of MNIST from a contaminated pool.

Builds a pool of N = 10,000 images with a fraction epsilon = 0.6 from K-MNIST or
Omniglot, audits 10% of it, fits the logistic-regression proxy on the audited
points (after flipping a fraction eta of their labels), and trains an IAF with
each loss below. Every model is evaluated on the clean MNIST test set.

  swd_contam             SWD on the pool                                   (Figure 3, Table 2)
  mmd_contam             naive MMD on the pool                             (Figure 3, Tables 2 and 5)
  ccmmd_contam           CC-MMD with the proxy scores                      (Figure 3, Tables 2 and 5)
  ccmmd_i1_contam        CC-MMD with I_tilde = 1 for every point           (Table 5)
  proxy_weighted_contam  MMD with the pool weighted by the proxy scores    (Table 5)
  proxy_filtered_contam  MMD on the points with proxy score >= 0.5         (Table 5)

Outputs, for run name R (default iaf_seed_{seed}_{contam}):
  results/R/{model}_metrics.json   test MMD^2 and FID of each model
  results/R/proxy_metrics.json     AUROC of the proxy scores on the pool
  checkpoints/R/{model}.pt         trained models (reused by later runs with identical settings)
  checkpoints/fid_net.pt           FID feature extractor (trained by the first run, then reused)

Usage:
    python experiments/exp4_iaf.py --seed 42 --contam-source kmnist
    python experiments/exp4_iaf.py --seed 42 --contam-source kmnist --proxy-label-noise 0 --proxy-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from torchvision.datasets import MNIST as RawMNIST

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exp4_iaf_config import CFG

from cc_mmd.data import sample_audit_subset
from cc_mmd.data.mnist import (
    build_contaminated_pool,
    compute_proxy_scores_mnist,
    dataset_to_tensor,
    load_datasets,
    load_omniglot,
    proxy_auroc,
)
from cc_mmd.evaluation.fid import extract_features, load_fid_net, train_fid_net
from cc_mmd.evaluation.metrics import evaluate_flow
from cc_mmd.losses.mmd import CCMMDLoss, NaiveMMDLoss, median_bandwidth
from cc_mmd.losses.swd import SWDLoss
from cc_mmd.models.flow import make_iaf
from cc_mmd.training.train import check_convergence, train_mmd
from cc_mmd.utils.checkpointing import find_matching_checkpoint, load_flow, save_metrics, save_model
from cc_mmd.utils.device import get_best_device

ALL_MODELS = [
    "swd_contam", "mmd_contam", "ccmmd_contam", "ccmmd_i1_contam",
    "proxy_weighted_contam", "proxy_filtered_contam",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 4: IAF trained on contaminated MNIST")
    p.add_argument("--device", default="auto",
                   help="'auto' picks the least busy GPU and enables TF32 (as in the paper's runs).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--contam-source", choices=["kmnist", "omniglot"], default="kmnist")
    p.add_argument("--models", default=",".join(ALL_MODELS),
                   help=f"Comma-separated subset of {', '.join(ALL_MODELS)} (default: all).")
    p.add_argument("--proxy-label-noise", type=float, default=None,
                   help=f"eta: fraction of audit labels flipped before fitting the proxy "
                        f"(default {CFG['label_noise']}).")
    p.add_argument("--proxy-only", action="store_true",
                   help="Fit the proxy, save and print its AUROC, and exit without training.")
    p.add_argument("--run-name", default=None,
                   help="Subdirectory of results/ and checkpoints/ for this run "
                        "(default iaf[_eta{eta}]_seed_{seed}_{contam}).")
    p.add_argument("--force-retrain", action="store_true",
                   help="Retrain even if a model trained with identical settings exists.")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--results-dir", default="results")
    return p.parse_args()


def default_run_name(arch: str, seed: int, contam: str, label_noise: float, default_noise: float) -> str:
    eta = "" if label_noise == default_noise else f"_eta{label_noise:g}"
    return f"{arch}{eta}_seed_{seed}_{contam}"


def main() -> None:
    args = parse_args()
    default_noise = CFG["label_noise"]
    if args.proxy_label_noise is not None:
        CFG["label_noise"] = args.proxy_label_noise
    run_name = args.run_name or default_run_name("iaf", args.seed, args.contam_source,
                                                 CFG["label_noise"], default_noise)

    device = get_best_device() if args.device == "auto" else torch.device(args.device)
    checkpoint_dir = Path(args.checkpoint_dir)
    results_dir = Path(args.results_dir) / run_name
    data_dir = Path(args.data_dir)

    selected = [m.strip() for m in args.models.split(",") if m.strip()]
    invalid = sorted(set(selected) - set(ALL_MODELS))
    if invalid:
        raise ValueError(f"Unknown model(s) {invalid} in --models; choices are {ALL_MODELS}")

    torch.manual_seed(args.seed)
    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed)

    epsilon = CFG["epsilon"]
    contam_label = {"kmnist": "K-MNIST", "omniglot": "Omniglot"}[args.contam_source]
    print(f"Experiment 4, IAF  |  epsilon = {epsilon:.0%}, contaminant = {contam_label}  |  "
          f"seed {args.seed}  |  eta = {CFG['label_noise']}  |  run {run_name}  |  device {device}")

    # ---- Data ---------------------------------------------------------------
    mnist_train, mnist_test, kmnist_train = load_datasets(data_dir)
    contam_train = load_omniglot(data_dir) if args.contam_source == "omniglot" else kmnist_train

    W_contam, I_true = build_contaminated_pool(
        mnist_train, contam_train, n_total=CFG["n_train"], epsilon=epsilon, rng=rng, device=device,
    )
    W_test = dataset_to_tensor(mnist_test, CFG["n_test"], rng=rng, device=device)

    # ---- Audit and proxy ----------------------------------------------------
    N = W_contam.shape[0]
    m = round(CFG["audit_fraction"] * N)
    L_indices = sample_audit_subset(N, m, rng=rng, device=device)
    I_true_L = I_true[L_indices]
    I_tilde = compute_proxy_scores_mnist(W_contam, L_indices, I_true_L, label_noise=CFG["label_noise"])
    auroc = proxy_auroc(I_tilde, I_true)
    print(f"Audit size m = {m}  |  proxy AUROC = {auroc:.4f}")
    save_metrics({"proxy_auroc": auroc, "label_noise": CFG["label_noise"]}, "proxy", results_dir)
    if args.proxy_only:
        return

    I_tilde_i1 = torch.ones(N, device=device)

    # ---- FID feature extractor ----------------------------------------------
    fid_ckpt = checkpoint_dir / "fid_net.pt"
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
    feat_test = extract_features(fid_net, test_images)
    sigma_test = median_bandwidth(W_test)

    # ---- Losses -------------------------------------------------------------
    # Kernel bandwidth: median distance among the audited MNIST points.
    clean_audit_idx = L_indices[I_true_L == 1]
    sigma = median_bandwidth(W_contam[clean_audit_idx])
    proxy_filtered_mask = I_tilde >= 0.5
    loss_builders = {
        "swd_contam":            lambda: SWDLoss(W_contam, n_projections=CFG["swd_projections"]),
        "mmd_contam":            lambda: NaiveMMDLoss(W_contam, sigma),
        "ccmmd_contam":          lambda: CCMMDLoss(W_contam, I_tilde, L_indices, I_true_L, sigma),
        "ccmmd_i1_contam":       lambda: CCMMDLoss(W_contam, I_tilde_i1, L_indices, I_true_L, sigma),
        # Proxy-weighted MMD is CC-MMD with the audit residuals set to zero.
        "proxy_weighted_contam": lambda: CCMMDLoss(W_contam, I_tilde, L_indices, I_tilde[L_indices], sigma),
        "proxy_filtered_contam": lambda: NaiveMMDLoss(W_contam[proxy_filtered_mask], sigma),
    }
    print(f"Kernel bandwidth sigma = {sigma:.4f} (from {clean_audit_idx.shape[0]} audited MNIST points)  |  "
          f"proxy-filtered subset: {int(proxy_filtered_mask.sum())} / {N}")

    def fresh_iaf() -> torch.nn.Module:
        return make_iaf(
            hidden_features=CFG["hidden_features"],
            num_layers=CFG["num_layers"],
            num_blocks_per_layer=CFG["num_blocks_per_layer"],
            dropout_prob=CFG["dropout_prob"],
            use_residual_blocks=CFG["use_residual_blocks"],
        ).to(device)

    # ---- Training -----------------------------------------------------------
    # Settings that identify a trained model; a checkpoint is reused only if they all match.
    arch_keys = ["hidden_features", "num_layers", "num_blocks_per_layer", "dropout_prob", "use_residual_blocks"]
    base_keys = ["n_train", "n_test", "epsilon"] + arch_keys + ["n_steps_mmd", "n_gen", "lr_mmd"]
    model_cfgs = {
        "swd_contam":            {k: CFG[k] for k in base_keys + ["swd_projections"]},
        "mmd_contam":            {k: CFG[k] for k in base_keys},
        "ccmmd_contam":          {k: CFG[k] for k in base_keys + ["audit_fraction", "label_noise"]},
        "ccmmd_i1_contam":       {k: CFG[k] for k in base_keys + ["audit_fraction"]},
        "proxy_weighted_contam": {k: CFG[k] for k in base_keys + ["audit_fraction", "label_noise"]},
        "proxy_filtered_contam": {k: CFG[k] for k in base_keys + ["label_noise"]},
    }
    for name, mode in [("ccmmd_i1_contam", "constant_one"), ("proxy_weighted_contam", "proxy_weighted"),
                       ("proxy_filtered_contam", "proxy_filtered")]:
        model_cfgs[name]["proxy_mode"] = mode
    for cfg in model_cfgs.values():
        cfg["seed"] = args.seed
        cfg["contam_source"] = args.contam_source

    trained = {}
    for i, name in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}] {name}")
        found = find_matching_checkpoint(name, model_cfgs[name], checkpoint_dir)
        if found and not args.force_retrain:
            print(f"  Found a matching checkpoint in {found}; skipping training.")
            trained[name] = load_flow(name, found, device, CFG)
        else:
            torch.manual_seed(args.seed)
            flow, hist = train_mmd(
                fresh_iaf(), loss_builders[name](),
                n_steps=CFG["n_steps_mmd"], n_gen=CFG["n_gen"], lr=CFG["lr_mmd"],
                device=device, desc=name,
            )
            check_convergence(hist, name=name)
            save_model(flow, name, checkpoint_dir / run_name, config=model_cfgs[name])
            trained[name] = flow

    # ---- Evaluation ---------------------------------------------------------
    for name, flow in trained.items():
        metrics = evaluate_flow(flow, W_test, sigma_test, fid_net, feat_test,
                                n_gen_eval=CFG["n_gen_eval"], n_eval_mmd=CFG["n_eval_mmd"], device=device)
        print(f"  {name:<22}  test MMD^2 = {metrics['test_mmd2']:.6f}  FID = {metrics['fid']:.2f}")
        save_metrics(metrics, name, results_dir)


if __name__ == "__main__":
    main()
