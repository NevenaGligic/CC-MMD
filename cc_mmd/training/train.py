"""Training loops for the generators of Experiment 4: MMD-type losses and the GAN baseline."""

from __future__ import annotations

import copy
from typing import Callable, List, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from cc_mmd.models.flow import flow_sample


def check_convergence(
    losses: List[float],
    name: str = "",
    window: float = 0.15,
    min_rel_improvement: float = 0.01,
) -> bool:
    """Print whether a recorded loss sequence has levelled off.

    Compares the mean of the last `window` fraction of the losses with the mean
    of the window before it.

    Args:
        losses: Loss values recorded during training.
        name: Label used in the printed summary.
        window: Fraction of the sequence in each comparison window.
        min_rel_improvement: Relative drop between the two windows below which
            training is considered converged.

    Returns:
        True if converged, False if the loss was still dropping.
    """
    n = len(losses)
    if n < 4:
        print(f"  [{name}] Too few loss values to assess convergence.")
        return True

    w = max(1, int(n * window))
    prev_mean = sum(losses[-(2 * w):-w]) / w
    last_mean = sum(losses[-w:]) / w
    rel_improvement = 0.0 if abs(prev_mean) < 1e-12 else (prev_mean - last_mean) / abs(prev_mean)
    converged = rel_improvement < min_rel_improvement

    tag = "CONVERGED" if converged else "STILL IMPROVING"
    print(f"  [{name}] {tag} - previous window avg: {prev_mean:.4f}  "
          f"last window avg: {last_mean:.4f}  relative improvement: {rel_improvement:.1%}")
    if last_mean > prev_mean:
        print(f"  [{name}] WARNING: loss increased in the final window ({last_mean - prev_mean:+.4f}).")
    return converged


def train_mmd_gen(
    model: nn.Module,
    sample_fn: Callable[[nn.Module, int, torch.device], torch.Tensor],
    loss_fn,
    n_steps: int,
    n_gen: int,
    lr: float,
    device: torch.device,
    desc: str = "MMD",
    max_grad_norm: float | None = 1.0,
    log_every: int = 100,
    early_stopping_patience: int = 15,
) -> Tuple[nn.Module, List[float]]:
    """Train a generator by minimising a sample-based loss (naive MMD, CC-MMD or SWD).

    Each step draws n_gen samples with sample_fn and takes an Adam step on
    loss_fn(samples). The learning rate follows cosine annealing from lr to
    lr / 100. The loss is averaged over windows of log_every steps; training
    stops early if the windowed loss has not improved for
    early_stopping_patience windows, and the best model is restored.

    Args:
        model: Generator on the target device.
        sample_fn: Callable (model, n, device) -> samples of shape (n, d).
        loss_fn: Loss object called on the generated samples.
        n_steps: Maximum number of gradient steps.
        n_gen: Number of generated samples per step.
        lr: Peak learning rate.
        device: Target device.
        desc: Progress-bar label.
        max_grad_norm: Gradient clipping norm (None to disable).
        log_every: Window length (in steps) for the recorded loss.
        early_stopping_patience: Number of windows without improvement before stopping.

    Returns:
        Tuple (model, step_losses): the best model in eval mode and the windowed losses.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=lr * 0.01)

    step_losses: List[float] = []
    best_loss = float("inf")
    best_state = None
    patience_counter = 0
    window_buf: List[float] = []

    model.train()
    for step in tqdm(range(n_steps), desc=desc, leave=False):
        X_gen = sample_fn(model, n_gen, device)
        loss = loss_fn(X_gen)
        if not torch.isfinite(loss):
            print(f"  Warning: non-finite loss at step {step}, skipping.")
            scheduler.step()
            continue
        optimizer.zero_grad()
        loss.backward()
        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        window_buf.append(loss.item())

        if (step + 1) % log_every == 0:
            avg_val = sum(window_buf) / len(window_buf)
            step_losses.append(avg_val)
            window_buf = []

            if avg_val < best_loss:
                best_loss = avg_val
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print(f"  Early stopping at step {step + 1} "
                      f"(no improvement for {early_stopping_patience} windows).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, step_losses


def train_mmd(
    flow: nn.Module,
    loss_fn,
    n_steps: int,
    n_gen: int,
    lr: float,
    device: torch.device,
    desc: str = "MMD",
    max_grad_norm: float | None = 1.0,
    log_every: int = 100,
    early_stopping_patience: int = 15,
) -> Tuple[nn.Module, List[float]]:
    """train_mmd_gen for an IAF built by make_iaf (samples with flow_sample)."""
    return train_mmd_gen(
        flow, flow_sample, loss_fn, n_steps, n_gen, lr, device,
        desc=desc, max_grad_norm=max_grad_norm, log_every=log_every,
        early_stopping_patience=early_stopping_patience,
    )


def train_gan(
    gen: nn.Module,
    disc: nn.Module,
    W: torch.Tensor,
    sample_fn: Callable[[nn.Module, int, torch.device], torch.Tensor],
    n_steps: int,
    batch_size: int,
    lr_g: float,
    lr_d: float,
    device: torch.device,
    n_disc_steps: int = 1,
    max_grad_norm: float | None = 1.0,
    log_every: int = 100,
    desc: str = "GAN",
) -> Tuple[nn.Module, List[float]]:
    """Train a generator adversarially with the non-saturating GAN objective.

    The discriminator is trained with BCEWithLogitsLoss to separate real pool
    samples from generated ones; the generator minimises -log D(G(z)). Both use
    Adam with betas (0.5, 0.999).

    Args:
        gen: Generator on the target device.
        disc: Discriminator on the target device.
        W: Training pool, shape (N, d).
        sample_fn: Callable (gen, n, device) -> samples of shape (n, d).
        n_steps: Number of generator updates.
        batch_size: Real and generated batch size per step.
        lr_g: Generator learning rate.
        lr_d: Discriminator learning rate.
        device: Target device.
        n_disc_steps: Discriminator updates per generator update.
        max_grad_norm: Gradient clipping norm for both networks (None to disable).
        log_every: Window length (in steps) for the recorded generator loss.
        desc: Progress-bar label.

    Returns:
        Tuple (gen, g_losses): the trained generator in eval mode and the windowed losses.
    """
    N = W.shape[0]
    criterion = nn.BCEWithLogitsLoss()
    opt_g = torch.optim.Adam(gen.parameters(), lr=lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr_d, betas=(0.5, 0.999))

    g_losses: List[float] = []
    window_buf: List[float] = []

    gen.train()
    disc.train()
    for step in tqdm(range(n_steps), desc=desc, leave=False):
        for _ in range(n_disc_steps):
            idx = torch.randint(0, N, (batch_size,), device=device)
            real = W[idx].detach()
            fake = sample_fn(gen, batch_size, device).detach()
            d_real = disc(real)
            d_fake = disc(fake)
            loss_d = (criterion(d_real, torch.ones_like(d_real))
                      + criterion(d_fake, torch.zeros_like(d_fake)))
            opt_d.zero_grad()
            loss_d.backward()
            if max_grad_norm is not None:
                nn.utils.clip_grad_norm_(disc.parameters(), max_grad_norm)
            opt_d.step()

        fake = sample_fn(gen, batch_size, device)
        d_fake = disc(fake)
        loss_g = criterion(d_fake, torch.ones_like(d_fake))
        opt_g.zero_grad()
        loss_g.backward()
        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(gen.parameters(), max_grad_norm)
        opt_g.step()
        window_buf.append(loss_g.item())

        if (step + 1) % log_every == 0:
            g_losses.append(sum(window_buf) / len(window_buf))
            window_buf = []

    gen.eval()
    return gen, g_losses
