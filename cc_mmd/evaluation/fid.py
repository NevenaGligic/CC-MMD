"""FID with a small CNN feature extractor trained on MNIST (Experiment 4).

InceptionV3 expects 299x299 RGB images, so FID is computed in the 128-dimensional
feature space of a small CNN classifier trained on clean MNIST instead. The
network is trained by the first run that needs it, saved, and reused by every
later run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import sqrtm
from torch import Tensor


class MNISTFeatureNet(nn.Module):
    """Conv(1->32, 3x3, stride 2) + ReLU -> Conv(32->64, 3x3, stride 2) + ReLU
    -> Linear(3136 -> 128) (features) -> Linear(128 -> 10) (class logits)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc_feat = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
        )
        self.classifier = nn.Linear(128, 10)

    def forward(self, x: Tensor) -> Tensor:
        """Class logits, shape (B, 10)."""
        return self.classifier(self.fc_feat(self.conv(x)))

    def features(self, x: Tensor) -> Tensor:
        """128-dimensional features, shape (B, 128)."""
        return self.fc_feat(self.conv(x))


def train_fid_net(
    train_images: Tensor,
    train_labels: Tensor,
    *,
    device: torch.device,
    n_epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    save_path: Path,
) -> MNISTFeatureNet:
    """Train the FID feature extractor on clean MNIST with cross-entropy and save it.

    Args:
        train_images: MNIST training images in [0, 1], shape (N, 1, 28, 28).
        train_labels: Class labels, shape (N,).
        device: Training device.
        n_epochs: Number of epochs.
        batch_size: Mini-batch size.
        lr: Adam learning rate.
        save_path: Where to save the state dict.

    Returns:
        Trained MNISTFeatureNet in eval mode.
    """
    net = MNISTFeatureNet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    x, y = train_images.to(device), train_labels.to(device)
    N = x.shape[0]

    net.train()
    for epoch in range(n_epochs):
        perm = torch.randperm(N, device=device)
        total_loss = 0.0
        for i in range(0, N, batch_size):
            idx = perm[i: i + batch_size]
            optimizer.zero_grad()
            loss = criterion(net(x[idx]), y[idx])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * idx.shape[0]
        print(f"  FID net  epoch {epoch + 1}/{n_epochs}  loss={total_loss / N:.4f}")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), save_path)
    print(f"  FID net saved to {save_path}")

    net.eval()
    return net


def load_fid_net(save_path: Path, *, device: torch.device) -> MNISTFeatureNet:
    """Load a saved FID feature extractor in eval mode."""
    net = MNISTFeatureNet().to(device)
    net.load_state_dict(torch.load(save_path, map_location=device))
    net.eval()
    return net


@torch.no_grad()
def extract_features(net: MNISTFeatureNet, images: Tensor, batch_size: int = 512) -> np.ndarray:
    """128-dimensional features of images in [0, 1] of shape (N, 1, 28, 28), as an (N, 128) array."""
    device = next(net.parameters()).device
    feats = []
    for i in range(0, images.shape[0], batch_size):
        feats.append(net.features(images[i: i + batch_size].to(device)).cpu().numpy())
    return np.concatenate(feats, axis=0)


def compute_fid(feat_real: np.ndarray, feat_gen: np.ndarray) -> float:
    """Frechet distance between Gaussians fitted to two feature sets.

    FID = ||mu_r - mu_g||^2 + Tr(Sigma_r + Sigma_g - 2 (Sigma_r Sigma_g)^(1/2)).
    """
    mu_r, mu_g = feat_real.mean(axis=0), feat_gen.mean(axis=0)
    sigma_r, sigma_g = np.cov(feat_real, rowvar=False), np.cov(feat_gen, rowvar=False)

    diff = mu_r - mu_g
    cov_prod = sigma_r @ sigma_g
    covmean = sqrtm(cov_prod + np.eye(cov_prod.shape[0]) * 1e-6)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff @ diff + np.trace(sigma_r + sigma_g - 2.0 * covmean))
