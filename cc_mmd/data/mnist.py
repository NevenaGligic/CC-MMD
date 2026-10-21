"""MNIST, K-MNIST and Omniglot loading, contaminated pools and proxy scores (Experiment 4).

All images are returned as 784-dimensional logit-space vectors (see transforms.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms as T
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch import Tensor
from torchvision.datasets import KMNIST, MNIST, Omniglot

from cc_mmd.data.transforms import get_mnist_transform


def load_omniglot(data_dir: str | Path) -> Omniglot:
    """Download (if needed) the Omniglot background set, resized to 28x28 and in logit space."""
    transform = T.Compose([T.Resize((28, 28))] + list(get_mnist_transform().transforms))
    return Omniglot(Path(data_dir), background=True, transform=transform, download=True)


def load_datasets(data_dir: str | Path) -> Tuple[MNIST, MNIST, KMNIST]:
    """Download (if needed) and return the MNIST train/test and K-MNIST train sets in logit space."""
    transform = get_mnist_transform()
    data_dir = Path(data_dir)
    mnist_train = MNIST(data_dir, train=True, transform=transform, download=True)
    mnist_test = MNIST(data_dir, train=False, transform=transform, download=True)
    kmnist_train = KMNIST(data_dir, train=True, transform=transform, download=True)
    return mnist_train, mnist_test, kmnist_train


def dataset_to_tensor(
    dataset,
    n: int,
    *,
    rng: torch.Generator,
    device: torch.device,
) -> Tensor:
    """Sample n images uniformly at random, without replacement, from a dataset.

    Args:
        dataset: Torchvision dataset whose items are (tensor_784, label).
        n: Number of images to sample.
        rng: PyTorch random generator.
        device: Target device.

    Returns:
        Tensor of shape (n, 784) in logit space.
    """
    # Dataset indexing happens on the CPU, so a CPU generator is seeded from rng.
    cpu_seed = torch.randint(0, 2**31 - 1, (1,), generator=rng, device=rng.device).item()
    cpu_rng = torch.Generator()
    cpu_rng.manual_seed(int(cpu_seed))
    indices = torch.randperm(len(dataset), generator=cpu_rng)[:n].tolist()
    images = torch.stack([dataset[i][0] for i in indices])
    return images.to(device)


def build_contaminated_pool(
    target_train,
    contam_train,
    n_total: int,
    epsilon: float,
    *,
    rng: torch.Generator,
    device: torch.device,
) -> Tuple[Tensor, Tensor]:
    """Build the contaminated pool W and its true membership indicators I.

    Draws round((1 - epsilon) * n_total) target images (MNIST) and the rest from
    the contaminating dataset (K-MNIST or Omniglot), then shuffles them together.

    Args:
        target_train: MNIST training set.
        contam_train: Contaminating training set.
        n_total: Pool size N.
        epsilon: Contamination rate in [0, 1).
        rng: PyTorch random generator.
        device: Target device.

    Returns:
        Tuple (W, I): pool of shape (N, 784) and indicators of shape (N,)
        (int32; 1 = MNIST, 0 = contaminant).
    """
    n_clean = round((1.0 - epsilon) * n_total)
    n_contam = n_total - n_clean

    W_clean = dataset_to_tensor(target_train, n_clean, rng=rng, device=device)
    W_contam = dataset_to_tensor(contam_train, n_contam, rng=rng, device=device)

    W = torch.cat([W_clean, W_contam], dim=0)
    I = torch.cat([
        torch.ones(n_clean, dtype=torch.int32, device=device),
        torch.zeros(n_contam, dtype=torch.int32, device=device),
    ])

    perm = torch.randperm(W.shape[0], generator=rng, device=device)
    return W[perm], I[perm]


def compute_proxy_scores_mnist(
    W: Tensor,
    L_indices: Tensor,
    I_true_L: Tensor,
    label_noise: float = 0.0,
    random_state: int = 0,
) -> Tensor:
    """Proxy scores from a logistic-regression classifier fitted on the audited subset.

    A fraction label_noise (eta in the paper) of the audit labels is flipped before
    fitting, to control proxy quality. The flipped labels are used only to fit the
    classifier; the CC-MMD correction always uses the true audit labels.

    Args:
        W: Contaminated pool in logit space, shape (N, 784).
        L_indices: Indices of the audited subset, shape (m,).
        I_true_L: True membership indicators of the audited points, shape (m,).
        label_noise: Fraction of audit labels flipped before fitting.
        random_state: Seed for the label flips and the classifier.

    Returns:
        Predicted probabilities of membership in [0, 1], shape (N,), on W's device.
    """
    rng = np.random.default_rng(random_state)

    W_np = W.cpu().numpy().astype(np.float32)
    L_np = L_indices.cpu().numpy()
    y_np = I_true_L.cpu().numpy().astype(int).copy()

    if label_noise > 0.0:
        n_flip = int(len(y_np) * label_noise)
        flip_idx = rng.choice(len(y_np), size=n_flip, replace=False)
        y_np[flip_idx] = 1 - y_np[flip_idx]

    clf = LogisticRegression(C=1.0, solver="saga", max_iter=1000, random_state=random_state)
    clf.fit(W_np[L_np], y_np)

    proba = clf.predict_proba(W_np)
    scores = proba[:, clf.classes_.tolist().index(1)].astype(np.float32)
    return torch.from_numpy(scores).to(W.device)


def proxy_auroc(I_tilde: Tensor, I_true: Tensor) -> float:
    """AUROC of the proxy scores against the true membership indicators of the whole pool."""
    return float(roc_auc_score(I_true.cpu().numpy().astype(int),
                               I_tilde.cpu().numpy().astype(np.float64)))
