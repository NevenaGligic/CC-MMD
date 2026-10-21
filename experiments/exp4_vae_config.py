"""Configuration of the VAE runs in Experiment 4.

Data, proxy, evaluation and distribution-matching training settings
(n_steps_mmd, n_gen, lr_mmd) are shared with the IAF runs (exp4_iaf_config.py).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp4_iaf_config import CFG

VAE_CFG = {
    **CFG,
    **dict(
        # ---- VAE architecture -----------------------------------------------
        vae_hidden_dim = 1024,
        vae_latent_dim = 128,

        # ---- ELBO training (VAE+MSE, VAE+Huber) -----------------------------
        vae_n_epochs    = 200,     # maximum epochs (early stopping on a validation split)
        vae_batch_size  = 256,
        vae_lr_elbo     = 1e-3,
        vae_huber_delta = 1.0,

        # ---- Distribution-matching training (VAE+MMD, VAE+CC-MMD) -----------
        vae_kl_batch_size = 256,   # pool points encoded per step for the KL term

        # ---- KL weight ------------------------------------------------------
        vae_beta_kl_elbo = 1.0,
        # MMD^2 on 784-dimensional logit-space data is of order 1e-2 to 1e-1 while the
        # KL term is of order 10 to 50, so a weight of 1 would reduce training to prior matching.
        vae_beta_kl_distrib = 0.01,
    )
}
