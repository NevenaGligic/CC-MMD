"""Inverse autoregressive flow (IAF) used in Experiment 4, built with nflows."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

DIM: int = 784   # flattened 28x28 image


def make_iaf(
    dim: int = DIM,
    hidden_features: int = 512,
    num_layers: int = 5,
    num_blocks_per_layer: int = 2,
    dropout_prob: float = 0.0,
    use_residual_blocks: bool = True,
    use_random_permutations: bool = True,
    use_random_masks: bool = False,
    batch_norm_within_layers: bool = True,
    batch_norm_between_layers: bool = True,
) -> nn.Module:
    """Build an inverse autoregressive flow (IAF).

    Each masked affine autoregressive transform (and each between-layer batch
    norm) is wrapped in InverseTransform, so sampling (z -> x) runs the
    autoregressive networks in their parallel direction and goes through
    BatchNorm.forward, which works in both training and eval mode. Only the
    sampling direction is used in this repository.

    Args:
        dim: Data dimensionality.
        hidden_features: Hidden width of each MADE conditioner.
        num_layers: Number of autoregressive layers.
        num_blocks_per_layer: Number of blocks in each MADE conditioner.
        dropout_prob: Dropout probability inside each MADE conditioner.
        use_residual_blocks: Use residual blocks in the MADE conditioners.
        use_random_permutations: Randomly permute dimensions between layers.
        use_random_masks: Use random MADE connectivity masks.
        batch_norm_within_layers: Batch normalization inside each MADE conditioner.
        batch_norm_between_layers: Batch normalization between consecutive layers.

    Returns:
        nflows Flow (on CPU).
    """
    from nflows.flows.base import Flow
    from nflows.transforms.base import CompositeTransform, InverseTransform
    from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
    from nflows.transforms.normalization import BatchNorm
    from nflows.transforms.permutations import RandomPermutation
    from nflows.distributions.normal import StandardNormal

    transforms = []
    for i in range(num_layers):
        transforms.append(InverseTransform(
            MaskedAffineAutoregressiveTransform(
                features=dim,
                hidden_features=hidden_features,
                num_blocks=num_blocks_per_layer,
                use_residual_blocks=use_residual_blocks,
                random_mask=use_random_masks,
                dropout_probability=dropout_prob,
                use_batch_norm=batch_norm_within_layers,
            )
        ))
        if i < num_layers - 1:
            if batch_norm_between_layers:
                transforms.append(InverseTransform(BatchNorm(dim)))
            if use_random_permutations:
                transforms.append(RandomPermutation(dim))

    return Flow(
        transform=CompositeTransform(transforms),
        distribution=StandardNormal([dim]),
    )


def flow_sample(flow: nn.Module, n: int, device: torch.device) -> Tensor:
    """Draw n samples x = T(z), z ~ N(0, I), keeping the graph for gradient-based training.

    Args:
        flow: IAF built by make_iaf.
        n: Number of samples.
        device: Target device.

    Returns:
        Samples of shape (n, dim).
    """
    dim = flow._distribution._shape[0]
    z = torch.randn(n, dim, device=device)
    x, _ = flow._transform.inverse(z)
    return x
