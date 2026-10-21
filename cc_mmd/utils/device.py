"""Device selection utilities."""

from __future__ import annotations

import subprocess

import torch


def _gpu_utilizations(num_gpus: int) -> list:
    """Query compute utilization (%) for each GPU via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            text=True,
        )
        vals = [int(x.strip()) for x in out.strip().splitlines()]
        if len(vals) == num_gpus:
            return vals
    except Exception:
        pass
    return [0] * num_gpus


def get_best_device() -> torch.device:
    """Select the best available compute device.

    Priority: CUDA (lowest utilisation, then most free memory) > MPS > CPU.
    """
    if torch.cuda.is_available():
        print("NVIDIA CUDA GPU found.")
        num_gpus = torch.cuda.device_count()
        if num_gpus == 1:
            print("Using single CUDA GPU: cuda:0")
            return torch.device("cuda:0")

        utils = _gpu_utilizations(num_gpus)
        best_idx = 0
        best_key = (float("inf"), 0)   # (utilisation, -free_mem) — lower is better
        for i in range(num_gpus):
            free, total = torch.cuda.mem_get_info(i)
            key = (utils[i], -free)
            print(f"  GPU {i}: util={utils[i]}%  {free / 1e9:.2f} GB free / {total / 1e9:.2f} GB total")
            if key < best_key:
                best_key = key
                best_idx = i

        print(f"Selected GPU {best_idx} (util={utils[best_idx]}%, {-best_key[1]/1e9:.2f} GB free).")
        return torch.device(f"cuda:{best_idx}")

    if torch.backends.mps.is_available():
        print("Apple Metal (MPS) found. Using mps.")
        return torch.device("mps")

    print("No GPU found. Using CPU.")
    return torch.device("cpu")
