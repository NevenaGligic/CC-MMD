"""Configuration of the IAF runs in Experiment 4, and the data, proxy and evaluation
settings shared with the VAE and GMMN runs and the ablations."""

CFG = dict(
    # ---- Data ---------------------------------------------------------------
    n_train        = 10_000,   # pool size N
    epsilon        = 0.60,     # contamination rate
    audit_fraction = 0.10,     # m / N
    n_test         = 10_000,   # clean MNIST test images used for evaluation

    # ---- IAF architecture ---------------------------------------------------
    hidden_features      = 512,
    num_layers           = 8,
    num_blocks_per_layer = 2,
    dropout_prob         = 0.0,
    use_residual_blocks  = True,

    # ---- Training of the distribution-matching losses -----------------------
    n_steps_mmd    = 10_000,   # maximum number of gradient steps
    n_gen          = 1_000,    # generated samples per step
    lr_mmd         = 3e-4,     # peak learning rate

    # ---- SWD baseline -------------------------------------------------------
    swd_projections = 200,

    # ---- Proxy --------------------------------------------------------------
    label_noise   = 0.25,      # eta: fraction of audit labels flipped before fitting the proxy

    # ---- Evaluation ---------------------------------------------------------
    n_gen_eval     = 50_000,   # generated samples for FID
    n_eval_mmd     =  5_000,   # generated samples for test MMD^2

    # ---- FID feature extractor ----------------------------------------------
    fid_epochs = 10,
)
