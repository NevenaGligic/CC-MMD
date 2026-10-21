"""Training loops for the VAE in Experiment 4.

train_vae_elbo     ELBO training (VAE+MSE, VAE+Huber): reconstruction loss + beta * KL,
                   epoch-based over the training pool.
train_vae_distrib  Distribution-matching training (VAE+MMD, VAE+CC-MMD): the MMD-type
                   loss of decoded prior samples + beta * KL on a mini-batch of the pool,
                   step-based.
"""

from __future__ import annotations

import copy
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from cc_mmd.models.vae import vae_sample


def _kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(N(mu, sigma^2) || N(0, I)), summed over latent dimensions and averaged over the batch."""
    return -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()


def train_vae_elbo(
    vae: nn.Module,
    W: torch.Tensor,
    n_epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    loss_type: str = "mse",
    beta_kl: float = 1.0,
    huber_delta: float = 1.0,
    max_grad_norm: Optional[float] = 5.0,
    val_fraction: float = 0.15,
    early_stopping_patience: int = 15,
) -> Tuple[nn.Module, List[float], List[float]]:
    """Train a VAE on the ELBO: reconstruction loss + beta_kl * KL.

    The reconstruction loss (MSE or Huber) is summed over dimensions and averaged
    over the batch. A random val_fraction of W is held out for early stopping, and
    the model with the lowest validation loss is restored. The learning rate
    follows cosine annealing from lr to lr / 100 over n_epochs.

    Args:
        vae: VAE on the target device.
        W: Training pool, shape (N, 784).
        n_epochs: Maximum number of epochs.
        batch_size: Mini-batch size.
        lr: Peak learning rate.
        device: Target device.
        loss_type: "mse" or "huber".
        beta_kl: Weight of the KL term.
        huber_delta: Huber transition point (ignored for MSE).
        max_grad_norm: Gradient clipping norm (None to disable).
        val_fraction: Fraction of W held out for validation.
        early_stopping_patience: Epochs without validation improvement before stopping.

    Returns:
        Tuple (vae, train_losses, val_losses): the best model in eval mode and per-epoch losses.
    """
    N = W.shape[0]
    n_val = max(1, int(N * val_fraction))
    n_train = N - n_val
    perm = torch.randperm(N, device=device)
    W_train = W[perm[:n_train]]
    W_val = W[perm[n_train:]]

    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.01)

    train_losses: List[float] = []
    val_losses: List[float] = []
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    pbar = tqdm(range(n_epochs), desc=f"VAE-ELBO ({loss_type})", leave=False)
    for epoch in pbar:
        vae.train()
        perm_t = torch.randperm(n_train, device=device)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, n_train, batch_size):
            idx = perm_t[i: i + batch_size]
            x = W_train[idx]
            x_hat, mu, logvar = vae(x)

            if loss_type == "mse":
                recon = F.mse_loss(x_hat, x, reduction="sum") / x.shape[0]
            else:
                recon = F.huber_loss(x_hat, x, reduction="sum", delta=huber_delta) / x.shape[0]
            loss = recon + beta_kl * _kl_loss(mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            if max_grad_norm is not None:
                nn.utils.clip_grad_norm_(vae.parameters(), max_grad_norm)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        train_losses.append(total_loss / n_batches)
        scheduler.step()

        vae.eval()
        with torch.no_grad():
            val_total, val_n = 0.0, 0
            for i in range(0, n_val, batch_size):
                x = W_val[i: i + batch_size]
                x_hat, mu, logvar = vae(x)
                if loss_type == "mse":
                    recon = F.mse_loss(x_hat, x, reduction="sum") / x.shape[0]
                else:
                    recon = F.huber_loss(x_hat, x, reduction="sum", delta=huber_delta) / x.shape[0]
                val_total += (recon + beta_kl * _kl_loss(mu, logvar)).item()
                val_n += 1
        avg_val = val_total / val_n
        val_losses.append(avg_val)

        pbar.set_postfix(train=f"{train_losses[-1]:.4f}", val=f"{avg_val:.4f}",
                         lr=f"{scheduler.get_last_lr()[0]:.2e}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = copy.deepcopy(vae.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f"  Early stopping after epoch {epoch + 1}.")
            break

    if best_state is not None:
        vae.load_state_dict(best_state)
    vae.eval()
    return vae, train_losses, val_losses


def train_vae_distrib(
    vae: nn.Module,
    loss_fn,
    W: torch.Tensor,
    n_steps: int,
    n_gen: int,
    lr: float,
    device: torch.device,
    beta_kl: float = 1.0,
    kl_batch_size: int = 256,
    desc: str = "VAE-distrib",
    max_grad_norm: Optional[float] = 1.0,
    log_every: int = 100,
    early_stopping_patience: int = 15,
) -> Tuple[nn.Module, List[float]]:
    """Train a VAE on a distribution-matching loss + beta_kl * KL.

    Each step decodes n_gen prior samples and evaluates loss_fn on them, and adds
    beta_kl times the KL term of a random mini-batch of kl_batch_size pool points
    passed through the encoder. Encoder and decoder are trained jointly. The
    learning rate follows cosine annealing from lr to lr / 100; early stopping on
    the windowed loss restores the best model, as in train_mmd_gen.

    Args:
        vae: VAE on the target device.
        loss_fn: NaiveMMDLoss or CCMMDLoss built on the pool.
        W: Training pool, shape (N, 784), used for the KL mini-batches.
        n_steps: Maximum number of gradient steps.
        n_gen: Number of generated samples per step.
        lr: Peak learning rate.
        device: Target device.
        beta_kl: Weight of the KL term.
        kl_batch_size: Number of pool points encoded per step for the KL term.
        desc: Progress-bar label.
        max_grad_norm: Gradient clipping norm (None to disable).
        log_every: Window length (in steps) for the recorded loss.
        early_stopping_patience: Number of windows without improvement before stopping.

    Returns:
        Tuple (vae, step_losses): the best model in eval mode and the windowed losses.
    """
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=lr * 0.01)

    N = W.shape[0]
    step_losses: List[float] = []
    best_loss = float("inf")
    best_state = None
    patience_counter = 0
    window_buf: List[float] = []

    vae.train()
    for step in tqdm(range(n_steps), desc=desc, leave=False):
        X_gen = vae_sample(vae, n_gen, device)
        dist_loss = loss_fn(X_gen)
        if not torch.isfinite(dist_loss):
            print(f"  Warning: non-finite distribution loss at step {step}, skipping.")
            scheduler.step()
            continue

        kl_idx = torch.randperm(N, device=device)[:kl_batch_size]
        mu, logvar = vae.encode(W[kl_idx])
        loss = dist_loss + beta_kl * _kl_loss(mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(vae.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        window_buf.append(loss.item())

        if (step + 1) % log_every == 0:
            avg_val = sum(window_buf) / len(window_buf)
            step_losses.append(avg_val)
            window_buf = []

            if avg_val < best_loss:
                best_loss = avg_val
                best_state = copy.deepcopy(vae.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print(f"  Early stopping at step {step + 1} "
                      f"(no improvement for {early_stopping_patience} windows).")
                break

    if best_state is not None:
        vae.load_state_dict(best_state)
    vae.eval()
    return vae, step_losses
