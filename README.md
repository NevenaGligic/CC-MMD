# Robust-by-Design Distributional Learning from Contaminated Samples

**Nevena Gligić, Arya Farahi**

*NeurIPS 2026*

> Modern data analysis pipelines increasingly rely on large datasets assembled through retrieval, scraping, automated filtering, and post-hoc curation, where the observed sample is often a contaminated mixture rather than the target population. Standard distributional objectives either ignore contamination and converge to the wrong target, or retain only a small trusted subset and discard most of the data. We study supervised and unsupervised learning when a clean reference sample is compared with a pooled sample from $r=(1-\epsilon)q+\epsilon z$, where $q$ is the latent target distribution and $z$ is an unknown contaminant. Motivated by settings with noisy membership scores for all observations and exact labels for a small audited subset, we propose contamination-corrected maximum mean discrepancy (CC-MMD), a design-based estimator that combines proxy scores with audited residual corrections to recover the oracle MMD that would have been computed from the latent clean sample. CC-MMD requires no calibration assumption on the proxy scores and no structural assumptions on the contaminant beyond standard kernel regularity. We prove finite-population design-unbiasedness of the corrected kernel sums, establish asymptotic normality of the audit correction, and show consistency for the target discrepancy $\mathrm{MMD}^2(p,q;k)$. Empirically, CC-MMD tracks the oracle under increasing contamination, improves parameter recovery and prediction in contaminated regression, and enables generative models trained on heavily corrupted mixtures to recover the clean target distribution. These results show that contaminated data need not be discarded or treated as the estimand. With sparse trusted labels and abundant noisy supervision, distributional objectives can be made robust by design.

---

## Installation

Python 3.10 or newer:

```bash
pip install -r requirements.txt
```

---

## Data

Experiments 1 and 2 simulate the data. 

The datasets for Experiments 3 and 4 are downloaded into `data/` on first use:

- **UCI Concrete Compressive Strength** (Experiment 3), from OpenML (dataset id 4353);
- **MNIST**, **K-MNIST** and **Omniglot** (Experiment 4), via `torchvision`.

---

## Reproducing the paper

Every script writes numbers to `results/` and figures to `figures/`, and prints the values reported in the text (including the proxy AUROCs). Seeds are fixed in the scripts, so the default commands reproduce the paper's settings.

| Paper | Command | Output |
|---|---|---|
| Figure&nbsp;1; Figure&nbsp;5 | `python experiments/exp1.py` | `figures/contamination_rate.pdf`<br>`figures/sim_visualization.pdf` |
| Table&nbsp;1; Figure&nbsp;6; Figure&nbsp;7 | `python experiments/exp2_regression.py` | `results/exp2_summary_200_table.{txt,tex}`<br>`figures/regression_realizations.pdf`<br>`figures/regression_summary_200.pdf` |
| Table&nbsp;3; Table&nbsp;4 | `python experiments/exp2_classifier_baselines.py` | `results/exp2_classifier_baselines_table{3,4}.{txt,tex}` |
| Figure&nbsp;2; Figure&nbsp;8; Figure&nbsp;9; Figure&nbsp;10 | `python experiments/exp3_concrete.py` | `figures/exp3_{mse,mmd}_vs_epsilon.pdf`<br>`figures/exp3_pool_distribution.pdf`<br>`figures/exp3_scatter.pdf`<br>`figures/exp3_qq.pdf` |
| Figure&nbsp;3; Tables&nbsp;2&nbsp;and&nbsp;5 | Experiment 4 runs as below, then `exp4_collect_results.py` and `exp4_plot.py` | `figures/exp4_{mmd2,fid}_{kmnist,omniglot}.pdf`<br>`results/exp4_table{2,5}.{txt,tex}` |
| Figure&nbsp;4 | `python experiments/exp4_ablation_epsilon.py` | `figures/exp4_ablation_epsilon_{mmd2,fid}.pdf` |
| Figure&nbsp;11 | `python experiments/exp4_ablation_proxy_quality.py` | `figures/exp4_ablation_proxy_quality_{mmd2,fid}.pdf` |
| Figure&nbsp;12 | `python experiments/exp4_ablation_audit_size.py` | `figures/exp4_ablation_audit_size_{mmd2,fid}.pdf` |



### Experiment 3: UCI Concrete regression

The figures can be redrawn from saved results with `python experiments/exp3_plot.py`.

### Experiment 4: generative modelling of MNIST

Each architecture is trained with all the loss functions for one seed and one contaminant per run (seeds 42 to 46, contaminants K-MNIST and Omniglot):

```bash
for seed in 42 43 44 45 46; do
  for contam in kmnist omniglot; do
    python experiments/exp4_iaf.py  --seed $seed --contam-source $contam
    python experiments/exp4_vae.py  --seed $seed --contam-source $contam
    python experiments/exp4_gmmn.py --seed $seed --contam-source $contam
  done
done
python experiments/exp4_collect_results.py   # Tables 2 and 5; prints the proxy AUROCs
python experiments/exp4_plot.py              # Figure 3
```

- The runs are independent and can be spread over several GPUs. `--models` restricts a run to some of the losses. 
- In Experiment 4 the default `--device auto` selects the least busy GPU available.
- The first Experiment 4 run trains the FID network and saves it to `checkpoints/fid_net.pt`; all later runs reuse that network, so FIDs are comparable across runs. 
- Trained Experiment 4 models are saved in `checkpoints/`. A later run with identical settings loads them instead of retraining (`--force-retrain` disables this).
- The settings of each architecture (pool and test sizes, network sizes, training steps) are in `exp4_iaf_config.py`, `exp4_vae_config.py` and `exp4_gmmn_config.py`.
- Each ablation script draws its figure at the end with `exp4_plot_ablation_epsilon.py`, `exp4_plot_ablation_proxy_quality.py` or `exp4_plot_ablation_audit_size.py`. Running one of these on its own redraws the figure from the saved results without retraining.

---

## Repository structure

```
CC-MMD/
├── cc_mmd/
│   ├── estimator.py          # oracle, naive, audit-only and CC-MMD estimators
│   ├── kernels.py            # Gaussian kernel
│   ├── data/                 # audit sampling, simulated data, MNIST/K-MNIST/Omniglot pools, proxy scores
│   ├── losses/               # MMD and CC-MMD training losses, SWD
│   ├── models/               # IAF, VAE, convolutional generator and discriminator
│   ├── training/             # training loops
│   ├── evaluation/           # test MMD^2 and FID
│   └── utils/                # checkpoints, device selection
├── experiments/
│   ├── exp1.py                            # Experiment 1
│   ├── exp2_regression.py                 # Experiment 2
│   ├── exp2_classifier_baselines.py       # Tables 3 and 4 (classifier-based baselines)
│   ├── exp3_concrete.py, exp3_plot.py     # Experiment 3
│   ├── exp4_{iaf,vae,gmmn}.py             # Experiment 4 training and evaluation
│   ├── exp4_{iaf,vae,gmmn}_config.py      # Experiment 4 settings
│   ├── exp4_collect_results.py            # Tables 2 and 5
│   ├── exp4_plot.py                       # Figure 3
│   ├── exp4_ablation_*.py                 # ablations (Figures 4, 11, 12)
│   └── exp4_plot_ablation_*.py            # ablation figures
├── results/                  # numerical results (created by the scripts)
├── requirements.txt
└── LICENSE
```

---

## Citation

```bibtex
@inproceedings{gligic2026robust,
  title     = {Robust-by-Design Distributional Learning from Contaminated Samples},
  author    = {Gligi{\'c}, Nevena and Farahi, Arya},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
