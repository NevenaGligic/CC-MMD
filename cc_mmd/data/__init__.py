"""Data generation, loading and the audit design shared by all experiments."""

import torch
from torch import Tensor


def sample_audit_subset(
    N: int,
    m: int,
    *,
    rng: torch.Generator,
    device: torch.device,
) -> Tensor:
    """Draw the audited subset L: a simple random sample of size m from {0, ..., N-1}.

    Args:
        N: Pool size.
        m: Audit size.
        rng: PyTorch random generator.
        device: Target device.

    Returns:
        Sorted index tensor of shape (m,).
    """
    perm = torch.randperm(N, generator=rng, device=device)
    return perm[:m].sort().values
