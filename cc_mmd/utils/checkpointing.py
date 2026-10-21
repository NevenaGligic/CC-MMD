"""Saving and loading of trained generators and evaluation metrics (Experiment 4).

Each trained model is saved as {name}.pt in a run directory, together with
{name}_config.json holding the settings it was trained with. Before training,
the scripts call find_matching_checkpoint to look for a model trained with
identical settings and load it instead of retraining.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from cc_mmd.models.flow import DIM, make_iaf
from cc_mmd.models.gmmn import make_gen
from cc_mmd.models.vae import make_vae


def save_config(config: dict, name: str, run_dir: Path) -> None:
    """Write the training settings of model `name` to run_dir/{name}_config.json."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / f"{name}_config.json", "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)


def find_matching_checkpoint(name: str, config: dict, base_dir: Path) -> Optional[Path]:
    """Return a run directory under base_dir holding {name}.pt trained with exactly `config`, or None."""
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return None
    for config_file in sorted(base_dir.glob(f"*/{name}_config.json")):
        run_dir = config_file.parent
        if not (run_dir / f"{name}.pt").exists():
            continue
        with open(config_file) as f:
            saved = json.load(f)
        if saved == config:
            return run_dir
    return None


def save_model(
    model: nn.Module,
    name: str,
    checkpoint_dir: Path,
    config: Optional[dict] = None,
) -> None:
    """Save a model's state dict to checkpoint_dir/{name}.pt and, if given, its training settings."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    local_path = checkpoint_dir / f"{name}.pt"
    torch.save(model.state_dict(), local_path)
    print(f"  Saved {name} to {local_path}")
    if config is not None:
        save_config(config, name, checkpoint_dir)


def save_metrics(metrics: dict, name: str, results_dir: Path) -> None:
    """Write an evaluation metrics dict to results_dir/{name}_metrics.json."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    local_path = results_dir / f"{name}_metrics.json"
    with open(local_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics {name} to {local_path}")


def _checkpoint_path(name: str, checkpoint_dir: Path) -> Path:
    local_path = Path(checkpoint_dir) / f"{name}.pt"
    if not local_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {local_path}")
    return local_path


def load_flow(name: str, checkpoint_dir: Path, device: torch.device, cfg: dict) -> nn.Module:
    """Load an IAF saved by save_model, rebuilt from the architecture settings in cfg."""
    local_path = _checkpoint_path(name, checkpoint_dir)
    flow = make_iaf(
        dim=DIM,
        hidden_features=cfg["hidden_features"],
        num_layers=cfg["num_layers"],
        num_blocks_per_layer=cfg["num_blocks_per_layer"],
        dropout_prob=cfg["dropout_prob"],
        use_residual_blocks=cfg["use_residual_blocks"],
    ).to(device)
    flow.load_state_dict(torch.load(local_path, map_location=device))
    flow.eval()
    print(f"  Loaded {name} from {local_path}")
    return flow


def load_vae(name: str, checkpoint_dir: Path, device: torch.device, cfg: dict) -> nn.Module:
    """Load a VAE saved by save_model, rebuilt from cfg["vae_hidden_dim"] and cfg["vae_latent_dim"]."""
    local_path = _checkpoint_path(name, checkpoint_dir)
    vae = make_vae(
        in_dim=DIM,
        hidden_dim=cfg["vae_hidden_dim"],
        latent_dim=cfg["vae_latent_dim"],
    ).to(device)
    vae.load_state_dict(torch.load(local_path, map_location=device))
    vae.eval()
    print(f"  Loaded {name} from {local_path}")
    return vae


def load_gen(name: str, checkpoint_dir: Path, device: torch.device, cfg: dict) -> nn.Module:
    """Load a ConvGenerator saved by save_model, rebuilt from cfg["gmmn_z_dim"]."""
    local_path = _checkpoint_path(name, checkpoint_dir)
    gen = make_gen(z_dim=cfg["gmmn_z_dim"]).to(device)
    gen.load_state_dict(torch.load(local_path, map_location=device))
    gen.eval()
    print(f"  Loaded {name} from {local_path}")
    return gen
