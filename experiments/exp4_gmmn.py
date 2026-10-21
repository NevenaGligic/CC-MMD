"""Experiment 4, GMMN: generative modelling of MNIST from a contaminated pool.

Same data, audit and proxy as exp4_iaf.py, with a convolutional generator trained by each of:

  gmmn_gan_contam             adversarial training (GAN baseline)          (Figure 3, Table 2)
  gmmn_mmd_contam             naive MMD                                    (Figure 3, Tables 2 and 5)
  gmmn_ccmmd_contam           CC-MMD                                       (Figure 3, Tables 2 and 5)
  gmmn_ccmmd_i1_contam        CC-MMD with I_tilde = 1                      (Table 5)
  gmmn_proxy_weighted_contam  proxy-weighted MMD                           (Table 5)
  gmmn_proxy_filtered_contam  MMD on points with proxy score >= 0.5        (Table 5)

Outputs, for run name R (default gmmn_seed_{seed}_{contam}):
  results/R/{model}_metrics.json, results/R/proxy_metrics.json,
  checkpoints/R/{model}.pt, checkpoints/fid_net.pt   (as in exp4_iaf.py)

Usage:
    python experiments/exp4_gmmn.py --seed 42 --contam-source kmnist
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from torchvision.datasets import MNIST as RawMNIST

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exp4_gmmn_config import GMMN_CFG
from exp4_iaf import default_run_name

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
from cc_mmd.evaluation.metrics import evaluate_gen
from cc_mmd.losses.mmd import CCMMDLoss, NaiveMMDLoss, median_bandwidth
from cc_mmd.models.gmmn import gen_sample, make_discriminator, make_gen
from cc_mmd.training.train import check_convergence, train_gan, train_mmd_gen
from cc_mmd.utils.checkpointing import find_matching_checkpoint, load_gen, save_metrics, save_model
from cc_mmd.utils.device import get_best_device

ALL_MODELS = [
    "gmmn_gan_contam", "gmmn_mmd_contam", "gmmn_ccmmd_contam", "gmmn_ccmmd_i1_contam",
    "gmmn_proxy_weighted_contam", "gmmn_proxy_filtered_contam",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 4: convolutional generator trained on contaminated MNIST")
    p.add_argument("--device", default="auto",
                   help="'auto' picks the least busy GPU and enables TF32 (as in the paper's runs).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--contam-source", choices=["kmnist", "omniglot"], default="kmnist")
    p.add_argument("--models", default=",".join(ALL_MODELS),
                   help=f"Comma-separated subset of {', '.join(ALL_MODELS)} (default: all).")
    p.add_argument("--proxy-label-noise", type=float, default=None,
                   help=f"eta: fraction of audit labels flipped before fitting the proxy "
                        f"(default {GMMN_CFG['label_noise']}).")
    p.add_argument("--run-name", default=None,
                   help="Subdirectory of results/ and checkpoints/ for this run "
                        "(default gmmn[_eta{eta}]_seed_{seed}_{contam}).")
    p.add_argument("--force-retrain", action="store_true",
                   help="Retrain even if a model trained with identical settings exists.")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--results-dir", default="results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    default_noise = GMMN_CFG["label_noise"]
    if args.proxy_label_noise is not None:
        GMMN_CFG["label_noise"] = args.proxy_label_noise
    run_name = args.run_name or default_run_name("gmmn", args.seed, args.contam_source,
                                                 GMMN_CFG["label_noise"], default_noise)

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

    epsilon = GMMN_CFG["epsilon"]
    contam_label = {"kmnist": "K-MNIST", "omniglot": "Omniglot"}[args.contam_source]
    print(f"Experiment 4, GMMN  |  epsilon = {epsilon:.0%}, contaminant = {contam_label}  |  "
          f"seed {args.seed}  |  eta = {GMMN_CFG['label_noise']}  |  run {run_name}  |  device {device}")

    # ---- Data ---------------------------------------------------------------
    mnist_train, mnist_test, kmnist_train = load_datasets(data_dir)
    contam_train = load_omniglot(data_dir) if args.contam_source == "omniglot" else kmnist_train

    W_contam, I_true = build_contaminated_pool(
        mnist_train, contam_train, n_total=GMMN_CFG["n_train"], epsilon=epsilon, rng=rng, device=device,
    )
    W_test = dataset_to_tensor(mnist_test, GMMN_CFG["n_test"], rng=rng, device=device)

    # ---- Audit and proxy ----------------------------------------------------
    N = W_contam.shape[0]
    m = round(GMMN_CFG["audit_fraction"] * N)
    L_indices = sample_audit_subset(N, m, rng=rng, device=device)
    I_true_L = I_true[L_indices]
    I_tilde = compute_proxy_scores_mnist(W_contam, L_indices, I_true_L, label_noise=GMMN_CFG["label_noise"])
    auroc = proxy_auroc(I_tilde, I_true)
    print(f"Audit size m = {m}  |  proxy AUROC = {auroc:.4f}")
    save_metrics({"proxy_auroc": auroc, "label_noise": GMMN_CFG["label_noise"]}, "proxy", results_dir)

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
                                n_epochs=GMMN_CFG["fid_epochs"], save_path=fid_ckpt)

    raw_test = RawMNIST(data_dir, train=False, transform=T.ToTensor(), download=False)
    test_images = torch.stack([raw_test[i][0] for i in range(GMMN_CFG["n_test"])])
    feat_test = extract_features(fid_net, test_images)
    sigma_test = median_bandwidth(W_test)

    # ---- Losses -------------------------------------------------------------
    # Kernel bandwidth: median distance among the audited MNIST points.
    clean_audit_idx = L_indices[I_true_L == 1]
    sigma = median_bandwidth(W_contam[clean_audit_idx])
    proxy_filtered_mask = I_tilde >= 0.5
    mmd_losses = {
        "gmmn_mmd_contam":            NaiveMMDLoss(W_contam, sigma),
        "gmmn_ccmmd_contam":          CCMMDLoss(W_contam, I_tilde, L_indices, I_true_L, sigma),
        "gmmn_ccmmd_i1_contam":       CCMMDLoss(W_contam, I_tilde_i1, L_indices, I_true_L, sigma),
        # Proxy-weighted MMD is CC-MMD with the audit residuals set to zero.
        "gmmn_proxy_weighted_contam": CCMMDLoss(W_contam, I_tilde, L_indices, I_tilde[L_indices], sigma),
        "gmmn_proxy_filtered_contam": NaiveMMDLoss(W_contam[proxy_filtered_mask], sigma),
    }
    print(f"Kernel bandwidth sigma = {sigma:.4f} (from {clean_audit_idx.shape[0]} audited MNIST points)  |  "
          f"proxy-filtered subset: {int(proxy_filtered_mask.sum())} / {N}")

    def fresh_gen() -> torch.nn.Module:
        return make_gen(z_dim=GMMN_CFG["gmmn_z_dim"]).to(device)

    def fresh_disc() -> torch.nn.Module:
        return make_discriminator(hidden_dim=GMMN_CFG["gan_disc_hidden"]).to(device)

    # ---- Training -----------------------------------------------------------
    # Settings that identify a trained model; a checkpoint is reused only if they all match.
    base_keys = ["n_train", "n_test", "epsilon", "gmmn_z_dim"]
    mmd_keys = base_keys + ["n_steps_mmd", "n_gen", "lr_mmd"]
    model_cfgs = {
        "gmmn_gan_contam":            {k: GMMN_CFG[k] for k in base_keys + ["gan_n_steps", "gan_batch_size",
                                                                            "gan_lr_g", "gan_lr_d", "gan_disc_hidden"]},
        "gmmn_mmd_contam":            {k: GMMN_CFG[k] for k in mmd_keys},
        "gmmn_ccmmd_contam":          {k: GMMN_CFG[k] for k in mmd_keys + ["audit_fraction", "label_noise"]},
        "gmmn_ccmmd_i1_contam":       {k: GMMN_CFG[k] for k in mmd_keys + ["audit_fraction"]},
        "gmmn_proxy_weighted_contam": {k: GMMN_CFG[k] for k in mmd_keys + ["audit_fraction", "label_noise"]},
        "gmmn_proxy_filtered_contam": {k: GMMN_CFG[k] for k in mmd_keys + ["label_noise"]},
    }
    for name, mode in [("gmmn_ccmmd_i1_contam", "constant_one"), ("gmmn_proxy_weighted_contam", "proxy_weighted"),
                       ("gmmn_proxy_filtered_contam", "proxy_filtered")]:
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
            trained[name] = load_gen(name, found, device, GMMN_CFG)
            continue

        torch.manual_seed(args.seed)
        if name == "gmmn_gan_contam":
            gen, _ = train_gan(
                fresh_gen(), fresh_disc(), W_contam, gen_sample,
                n_steps=GMMN_CFG["gan_n_steps"], batch_size=GMMN_CFG["gan_batch_size"],
                lr_g=GMMN_CFG["gan_lr_g"], lr_d=GMMN_CFG["gan_lr_d"], device=device, desc=name,
            )
        else:
            gen, hist = train_mmd_gen(
                fresh_gen(), gen_sample, mmd_losses[name],
                n_steps=GMMN_CFG["n_steps_mmd"], n_gen=GMMN_CFG["n_gen"], lr=GMMN_CFG["lr_mmd"],
                device=device, desc=name,
            )
            check_convergence(hist, name=name)
        save_model(gen, name, checkpoint_dir / run_name, config=model_cfgs[name])
        trained[name] = gen

    # ---- Evaluation ---------------------------------------------------------
    for name, gen in trained.items():
        metrics = evaluate_gen(gen, gen_sample, W_test, sigma_test, fid_net, feat_test,
                               n_gen_eval=GMMN_CFG["n_gen_eval"], n_eval_mmd=GMMN_CFG["n_eval_mmd"],
                               device=device)
        print(f"  {name:<27}  test MMD^2 = {metrics['test_mmd2']:.6f}  FID = {metrics['fid']:.2f}")
        save_metrics(metrics, name, results_dir)


if __name__ == "__main__":
    main()
