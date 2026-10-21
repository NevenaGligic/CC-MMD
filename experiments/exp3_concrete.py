"""Experiment 3: MLP regression on contaminated UCI Concrete data (Figure 2, Appendix).

Target q: concrete cured for 28 days. Contaminants z: concrete cured for at most
7 days. The curing age is removed from the features. For each contamination
rate epsilon in {0.0, ..., 0.6}, 300 target samples are pooled with
contaminants, 20% of the pool is audited, and an MLP is trained with seven
objectives: Oracle (MSE on the true targets), Audit-only (MSE on the audited
targets), MSE, Huber, Tukey, naive MMD and CC-MMD. All are evaluated on 125
held-out target samples. The proxy is a random forest fitted on the audited
points after flipping a fraction eta of their labels.

Replicate r uses seed --seed + 1000 * r. The scatter, QQ and pool figures show
the first replicate at epsilon = 0.5 (seed 42).

Outputs:
  results/exp3_results.json   per-replicate test MSE, test MMD^2 and proxy AUROC,
                              plus the predictions of the replicate shown in the
                              scatter, QQ and pool figures
  figures/exp3_*.pdf          the five figures drawn by exp3_plot.py

Usage:
    python experiments/exp3_concrete.py                          # eta = 0.2, as in the paper
    python experiments/exp3_concrete.py --proxy-label-noise 0.0  # eta = 0 reference AUROCs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cc_mmd.data import sample_audit_subset
from cc_mmd.losses.mmd import CCMMDLoss, NaiveMMDLoss, median_bandwidth
from exp3_plot import METHODS, load_results, plot_all

CFG = dict(
    N_train           = 300,     # target samples in the training pool
    N_test            = 125,     # held-out target samples
    audit_fraction    = 0.20,    # m = round(audit_fraction * pool size)
    proxy_label_noise = 0.20,    # eta: fraction of audit labels flipped before fitting the proxy
    sigma_noise       = 0.05,    # std of the noise added to MLP outputs in the MMD objectives
    hidden            = [64, 64, 32],
    lr                = 1e-3,
    weight_decay      = 1e-4,
    n_epochs_fast     = 300,     # Oracle, Audit-only, MSE, Huber, Tukey
    n_epochs_warmup   = 50,      # MSE warm-up before Tukey; Huber warm-up before the MMD objectives
    n_epochs_mmd      = 400,     # naive MMD and CC-MMD (after the warm-up)
    batch_size        = 64,      # pointwise losses; the MMD objectives use the full pool
    tukey_c           = 4.685,
    epsilons          = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    epsilon_demo      = 0.5,     # epsilon of the scatter, QQ and pool figures
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_natural_concrete(cache_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """UCI Concrete (OpenML id 4353) split by curing age, with the age column removed.

    Returns (X_q, Y_q, X_z, Y_z): 28-day samples (target) and samples cured for
    at most 7 days (contaminants). The dataset is cached in cache_dir/concrete.npz.
    """
    cache_path = cache_dir / "concrete.npz"
    if not cache_path.exists():
        print("  Downloading Concrete Compressive Strength dataset (OpenML id 4353) ...")
        ds = fetch_openml(data_id=4353, as_frame=True, parser="auto")
        df = ds.frame
        target_col = ds.target_names[0] if ds.target_names else df.columns[-1]
        feat_cols = [c for c in df.columns if c != target_col]
        X = df[feat_cols].values.astype(np.float32)
        Y = df[target_col].values.astype(np.float32)
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, X=X, Y=Y)
    else:
        d = np.load(cache_path)
        X, Y = d["X"].astype(np.float32), d["Y"].astype(np.float32)

    age_col = 7
    inlier_mask = X[:, age_col] == 28.0
    contam_mask = X[:, age_col] <= 7.0

    X_q = np.delete(X[inlier_mask], age_col, axis=1)
    X_z = np.delete(X[contam_mask], age_col, axis=1)
    return X_q, Y[inlier_mask], X_z, Y[contam_mask]


def split_and_scale(
    X_q: np.ndarray, Y_q: np.ndarray,
    X_z: np.ndarray, Y_z: np.ndarray,
    N_train: int, N_test: int, seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split the target data into train/test and standardise everything with scalers
    fitted on the target training data only.

    Returns (X_q_train, Y_q_train, X_z, Y_z, X_test, Y_test), all standardised.
    """
    rng = np.random.default_rng(seed)
    idx_q = rng.permutation(len(Y_q))
    X_q_tr, Y_q_tr = X_q[idx_q[:N_train]], Y_q[idx_q[:N_train]]
    X_te, Y_te = X_q[idx_q[N_train: N_train + N_test]], Y_q[idx_q[N_train: N_train + N_test]]

    xs, ys = StandardScaler(), StandardScaler()
    X_q_train = xs.fit_transform(X_q_tr).astype(np.float32)
    Y_q_train = ys.fit_transform(Y_q_tr.reshape(-1, 1)).ravel().astype(np.float32)
    X_test = xs.transform(X_te).astype(np.float32)
    Y_test = ys.transform(Y_te.reshape(-1, 1)).ravel().astype(np.float32)
    X_z_scaled = xs.transform(X_z).astype(np.float32)
    Y_z_scaled = ys.transform(Y_z.reshape(-1, 1)).ravel().astype(np.float32)

    return X_q_train, Y_q_train, X_z_scaled, Y_z_scaled, X_test, Y_test


def make_contaminated_pool(
    X_inlier: np.ndarray, Y_inlier: np.ndarray,
    X_contam: np.ndarray, Y_contam: np.ndarray,
    epsilon: float, seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All target training samples plus floor(N epsilon / (1 - epsilon)) contaminants
    drawn with replacement, shuffled. Returns (X_pool, Y_pool, I) with I = 1 for targets."""
    rng = np.random.default_rng(seed)
    n_inlier = len(Y_inlier)

    if epsilon == 0.0:
        return X_inlier.copy(), Y_inlier.copy(), np.ones(n_inlier, dtype=np.float32)

    n_contam = int(n_inlier * (epsilon / (1.0 - epsilon)))
    contam_idx = rng.choice(len(Y_contam), n_contam, replace=True)

    X_pool = np.vstack([X_inlier.copy(), X_contam[contam_idx]])
    Y_pool = np.concatenate([Y_inlier.copy(), Y_contam[contam_idx]])
    I = np.concatenate([np.ones(n_inlier, dtype=np.float32), np.zeros(n_contam, dtype=np.float32)])

    perm = rng.permutation(n_inlier + n_contam)
    return X_pool[perm], Y_pool[perm], I[perm]


# ---------------------------------------------------------------------------
# Proxy scores and evaluation
# ---------------------------------------------------------------------------

def compute_proxy_scores(
    X_pool: np.ndarray,
    Y_pool: np.ndarray,
    L_indices: np.ndarray,
    I_true_L: np.ndarray,
    label_noise: float = 0.0,
    random_state: int = 0,
) -> np.ndarray:
    """Random-forest proxy scores from the joint (X, Y) features of the audited points.

    A fraction label_noise of the audit labels is flipped before fitting (the
    flipped labels are used only here). Returns 0.5 for every point if the
    (possibly flipped) audit labels contain a single class.
    """
    rng = np.random.default_rng(random_state)

    y_audit = I_true_L.astype(int).copy()
    if label_noise > 0.0:
        n_flip = int(len(y_audit) * label_noise)
        flip_idx = rng.choice(len(y_audit), size=n_flip, replace=False)
        y_audit[flip_idx] = 1 - y_audit[flip_idx]

    if len(np.unique(y_audit)) < 2:
        return np.full(len(Y_pool), 0.5, dtype=np.float32)

    XY_pool = np.column_stack([X_pool, Y_pool])
    clf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=random_state)
    clf.fit(XY_pool[L_indices], y_audit)

    proba = clf.predict_proba(XY_pool)
    return proba[:, clf.classes_.tolist().index(1)].astype(np.float32)


def proxy_auroc(I_tilde: np.ndarray, I_true: np.ndarray) -> float:
    """AUROC of the proxy scores on the whole pool (NaN when the pool has no contaminants)."""
    if len(np.unique(I_true)) < 2:
        return float("nan")
    return float(roc_auc_score(I_true.astype(int), I_tilde.astype(np.float64)))


def compute_mmd_eval(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Unbiased MMD^2 between true and predicted test outputs (1-D Gaussian kernel,
    median-heuristic bandwidth on the combined sample)."""
    x = torch.tensor(y_true, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(y_pred, dtype=torch.float32).unsqueeze(1)
    sigma = float(median_bandwidth(torch.cat([x, y], dim=0)))
    if sigma < 1e-8:
        sigma = 1.0

    def _k(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        diff = a.unsqueeze(1) - b.unsqueeze(0)
        return torch.exp(-diff.pow(2).squeeze(-1) / (2 * sigma ** 2))

    n, m = len(x), len(y)
    Kxx, Kyy, Kxy = _k(x, x), _k(y, y), _k(x, y)
    kxx = (Kxx.sum() - Kxx.diag().sum()) / (n * (n - 1))
    kyy = (Kyy.sum() - Kyy.diag().sum()) / (m * (m - 1))
    return float(kxx + kyy - 2 * Kxy.sum() / (n * m))


# ---------------------------------------------------------------------------
# Model and losses
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int]) -> None:
        super().__init__()
        layers: list = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _fresh_mlp(cfg: dict, in_dim: int, device: torch.device, seed: int) -> MLP:
    """New MLP after resetting the random state to `seed`, so every method starts identically."""
    torch.manual_seed(seed)
    return MLP(in_dim, cfg["hidden"]).to(device)


def _huber_loss(y_pred: torch.Tensor, y_true: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    r = y_true - y_pred
    return torch.where(r.abs() <= delta, 0.5 * r.pow(2), delta * (r.abs() - 0.5 * delta)).mean()


def _tukey_rho_loss(y_pred: torch.Tensor, y_true: torch.Tensor, c: float = 4.685) -> torch.Tensor:
    """Tukey biweight loss with a MAD scale computed from the current residuals."""
    residuals = y_true - y_pred
    with torch.no_grad():
        med = residuals.median()
        mad = (residuals - med).abs().median().clamp(min=1e-6)
        scale = 1.4826 * mad
    u = residuals / (scale * c)
    c_eff = (scale * c).detach()
    return torch.where(u.abs() <= 1.0,
                       c_eff ** 2 / 6.0 * (1.0 - (1.0 - u.pow(2)).pow(3)),
                       torch.full_like(u, c_eff.item() ** 2 / 6.0)).mean()


def _evaluate(mlp: MLP, X_t: torch.Tensor, Y_t: torch.Tensor) -> float:
    mlp.eval()
    with torch.no_grad():
        return float(((mlp(X_t) - Y_t) ** 2).mean())


def _train_supervised(mlp: MLP, X: torch.Tensor, Y: torch.Tensor, loss_fn, cfg: dict) -> MLP:
    """Mini-batch Adam on a pointwise loss for cfg["n_epochs_fast"] epochs."""
    opt = torch.optim.Adam(mlp.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    N = X.shape[0]
    bs = min(cfg["batch_size"], N)

    for _ in range(cfg["n_epochs_fast"]):
        mlp.train()
        perm = torch.randperm(N, device=X.device)
        for start in range(0, N, bs):
            idx = perm[start: start + bs]
            opt.zero_grad()
            loss_fn(mlp(X[idx]), Y[idx]).backward()
            opt.step()
    return mlp


def _train_mmd_obj(mlp: MLP, X_feat: torch.Tensor, loss_obj, cfg: dict) -> MLP:
    """Full-batch Adam on an MMD-type objective between the model sample
    [X, (mlp(X) + noise) * y_scale] and the pool held by loss_obj."""
    opt = torch.optim.Adam(mlp.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    y_scale = cfg["y_scale"]

    for _ in range(cfg["n_epochs_mmd"]):
        mlp.train()
        opt.zero_grad()
        y_hat = mlp(X_feat)
        y_noisy = y_hat + (torch.randn_like(y_hat) * cfg["sigma_noise"]).detach()
        M = torch.cat([X_feat, (y_noisy * y_scale).unsqueeze(1)], dim=1)
        loss = loss_obj(M)
        if not torch.isfinite(loss):
            continue
        loss.backward()
        opt.step()
    return mlp


def _to_t(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(arr, device=device)


def _build_joint_W(X_t: torch.Tensor, Y_t: torch.Tensor, y_scale: float) -> torch.Tensor:
    """Joint (X, y_scale * Y) matrix on which the MMD kernels act."""
    return torch.cat([X_t, (Y_t * y_scale).unsqueeze(1)], dim=1)


# ---------------------------------------------------------------------------
# The seven methods
# ---------------------------------------------------------------------------

def train_pointwise(X: np.ndarray, Y: np.ndarray, loss_fn, cfg, device, seed, warmup=False) -> MLP:
    """MSE / Huber / Tukey on (X, Y); Tukey is preceded by an MSE warm-up."""
    mlp = _fresh_mlp(cfg, X.shape[1], device, seed)
    X_t, Y_t = _to_t(X, device), _to_t(Y, device)
    if warmup:
        mlp = _train_supervised(mlp, X_t, Y_t, F.mse_loss, {**cfg, "n_epochs_fast": cfg["n_epochs_warmup"]})
    return _train_supervised(mlp, X_t, Y_t, loss_fn, cfg)


def train_mmd(X_pool: np.ndarray, Y_pool: np.ndarray, make_loss, cfg, device, seed) -> MLP:
    """Huber warm-up followed by an MMD-type objective with bandwidth median / D."""
    mlp = _fresh_mlp(cfg, X_pool.shape[1], device, seed)
    X_t, Y_t = _to_t(X_pool, device), _to_t(Y_pool, device)
    mlp = _train_supervised(mlp, X_t, Y_t, _huber_loss, {**cfg, "n_epochs_fast": cfg["n_epochs_warmup"]})
    W = _build_joint_W(X_t, Y_t, cfg["y_scale"])
    sigma = median_bandwidth(W) / W.shape[1]
    return _train_mmd_obj(mlp, X_t, make_loss(W, sigma), cfg)


def run_realization(
    X_clean: np.ndarray, Y_clean: np.ndarray,
    X_contam: np.ndarray, Y_contam: np.ndarray,
    X_test: np.ndarray, Y_test: np.ndarray,
    epsilon: float, cfg: dict, seed: int, device: torch.device,
) -> dict:
    """Build one contaminated pool, train the seven methods (each from the same
    initialization) and evaluate them on the test set."""
    X_pool, Y_pool, I_true = make_contaminated_pool(X_clean, Y_clean, X_contam, Y_contam, epsilon, seed)

    N = len(Y_pool)
    m = max(2, round(cfg["audit_fraction"] * N))
    rng_t = torch.Generator()
    rng_t.manual_seed(seed)
    L_np = sample_audit_subset(N, m, rng=rng_t, device=torch.device("cpu")).numpy()
    I_true_L = I_true[L_np]
    I_tilde = compute_proxy_scores(X_pool, Y_pool, L_np, I_true_L,
                                   label_noise=cfg["proxy_label_noise"], random_state=seed)

    inlier = I_true.astype(bool)
    audit_inliers = L_np[I_true_L.astype(bool)]
    if len(audit_inliers) == 0:
        audit_inliers = L_np

    I_til_t = _to_t(I_tilde, device)
    L_idx_t = torch.tensor(L_np, dtype=torch.long, device=device)
    I_true_t = _to_t(I_true_L, device)

    mlps = dict(
        oracle=train_pointwise(X_pool[inlier], Y_pool[inlier], F.mse_loss, cfg, device, seed),
        audit_only=train_pointwise(X_pool[audit_inliers], Y_pool[audit_inliers], F.mse_loss, cfg, device, seed),
        mse=train_pointwise(X_pool, Y_pool, F.mse_loss, cfg, device, seed),
        huber=train_pointwise(X_pool, Y_pool, _huber_loss, cfg, device, seed),
        tukey=train_pointwise(X_pool, Y_pool, lambda yp, yt: _tukey_rho_loss(yp, yt, cfg["tukey_c"]),
                              cfg, device, seed, warmup=True),
        naive_mmd=train_mmd(X_pool, Y_pool, lambda W, s: NaiveMMDLoss(W, s), cfg, device, seed),
        cc_mmd=train_mmd(X_pool, Y_pool, lambda W, s: CCMMDLoss(W, I_til_t, L_idx_t, I_true_t, s),
                         cfg, device, seed),
    )

    X_te, Y_te = _to_t(X_test, device), _to_t(Y_test, device)
    preds = {}
    for k, mlp in mlps.items():
        mlp.eval()
        with torch.no_grad():
            preds[k] = mlp(X_te).cpu().numpy()

    return dict(
        epsilon=float(epsilon),
        seed=int(seed),
        m_audit=int(m),
        proxy_auroc=proxy_auroc(I_tilde, I_true),
        Y_pool=Y_pool,
        I_true=I_true,
        mse={k: _evaluate(mlp, X_te, Y_te) for k, mlp in mlps.items()},
        mmd={k: compute_mmd_eval(Y_test, preds[k]) for k in mlps},
        preds=preds,
    )


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def _to_json_safe(obj):
    """Convert numpy types to JSON types (NaN becomes null)."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        obj = float(obj)
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, float) and obj != obj:
        return None
    return obj


def save_results_json(all_results: dict, demo: dict, cfg: dict, Y_test: np.ndarray, out_path: Path) -> None:
    payload = {
        "cfg": _to_json_safe(cfg),
        "Y_test": _to_json_safe(Y_test),
        "sweep": {
            str(eps): [{k: _to_json_safe(r[k]) for k in ("seed", "m_audit", "proxy_auroc", "mse", "mmd")}
                       for r in reps]
            for eps, reps in all_results.items()
        },
        "demo": {k: _to_json_safe(demo[k]) for k in
                 ("epsilon", "seed", "m_audit", "proxy_auroc", "mse", "mmd", "preds", "Y_pool", "I_true")},
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved {out_path}")


def print_summary(all_results: dict, demo: dict, epsilons: list) -> None:
    """Proxy AUROC (mean +/- sample std) and test MSE / MMD^2 (mean +/- std) per epsilon."""
    print("\nProxy AUROC against the true pool membership (mean +/- std over replicates):")
    for eps in epsilons:
        if eps > 0:
            a = np.array([r["proxy_auroc"] for r in all_results[eps]])
            print(f"  epsilon = {eps:.1f}: {a.mean():.3f} +/- {a.std(ddof=1):.3f}")
    print(f"  replicate shown in the scatter/QQ figures (epsilon = {demo['epsilon']}, seed {demo['seed']}): "
          f"{demo['proxy_auroc']:.4f}")

    for metric, label in [("mse", "test MSE"), ("mmd", "test MMD^2")]:
        print(f"\nMean +/- std {label} over replicates:")
        print(f"  {'eps':>4}  " + "  ".join(f"{k:>17}" for k in METHODS))
        for eps in epsilons:
            cells = []
            for k in METHODS:
                v = np.array([r[metric][k] for r in all_results[eps]])
                cells.append(f"{v.mean():>8.4f} +-{v.std():<7.4f}")
            print(f"  {eps:>4.1f}  " + "  ".join(cells))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 3: MLP regression on contaminated UCI Concrete data")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42, help="Base seed; replicate r uses seed + 1000 r.")
    p.add_argument("--n-reps", type=int, default=20, help="Replicates per contamination rate.")
    p.add_argument("--audit-fraction", type=float, default=CFG["audit_fraction"])
    p.add_argument("--proxy-label-noise", type=float, default=CFG["proxy_label_noise"],
                   help="eta: fraction of audit labels flipped before fitting the proxy.")
    p.add_argument("--epsilon-demo", type=float, default=CFG["epsilon_demo"],
                   help="Contamination rate shown in the scatter, QQ and pool figures.")
    p.add_argument("--figures-dir", default="figures")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--data-dir", default="data")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    CFG["audit_fraction"] = args.audit_fraction
    CFG["proxy_label_noise"] = args.proxy_label_noise
    CFG["epsilon_demo"] = args.epsilon_demo
    if CFG["epsilon_demo"] not in CFG["epsilons"]:
        raise ValueError(f"--epsilon-demo must be one of {CFG['epsilons']}")

    out_dir = Path(args.figures_dir)
    res_dir = Path(args.results_dir)
    data_dir = Path(args.data_dir)
    res_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}  |  Replicates: {args.n_reps}  |  Seed: {args.seed}  |  "
          f"audit fraction: {CFG['audit_fraction']}  |  eta: {CFG['proxy_label_noise']}")

    X_q, Y_q, X_z, Y_z = load_natural_concrete(data_dir)
    print(f"Target samples: {len(Y_q)}  |  Contaminant samples: {len(Y_z)}")
    X_clean, Y_clean, X_contam, Y_contam, X_test, Y_test = split_and_scale(
        X_q, Y_q, X_z, Y_z, CFG["N_train"], CFG["N_test"], seed=args.seed,
    )
    # Scaling Y by sqrt(number of features) balances it against the features in the kernel.
    CFG["y_scale"] = float(np.sqrt(X_clean.shape[1]))

    all_results: dict = {eps: [] for eps in CFG["epsilons"]}
    for eps in CFG["epsilons"]:
        for rep in tqdm(range(args.n_reps), desc=f"eps={eps:.1f}"):
            all_results[eps].append(run_realization(
                X_clean, Y_clean, X_contam, Y_contam, X_test, Y_test,
                epsilon=eps, cfg=CFG, seed=args.seed + rep * 1000, device=device,
            ))

    # The scatter, QQ and pool figures show the first replicate at epsilon_demo.
    demo = all_results[CFG["epsilon_demo"]][0]
    json_path = res_dir / "exp3_results.json"
    save_results_json(all_results, demo, CFG, Y_test, json_path)
    print_summary(all_results, demo, CFG["epsilons"])

    plot_all(load_results(json_path), out_dir)
    print("\nExperiment 3 complete.")


if __name__ == "__main__":
    main()
