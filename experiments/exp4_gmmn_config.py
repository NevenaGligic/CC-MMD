"""Configuration of the GMMN (convolutional generator) and GAN runs in Experiment 4.

Data, proxy, evaluation and distribution-matching training settings are shared
with the IAF runs (exp4_iaf_config.py).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp4_iaf_config import CFG

GMMN_CFG = {
    **CFG,
    **dict(
        # ---- Generator architecture ----------------------------------------
        gmmn_z_dim = 128,          # noise dimensionality

        # ---- GAN baseline ---------------------------------------------------
        gan_n_steps     = 30_000,
        gan_batch_size  = 500,     # real and generated samples per step
        gan_lr_g        = 2e-4,
        gan_lr_d        = 5e-5,
        gan_disc_hidden = 512,
    )
}
