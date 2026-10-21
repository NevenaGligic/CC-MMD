"""Experiment 4: collect the per-run metrics, write Tables 2 and 5 and print the proxy AUROCs.

Scans results/ for run directories named {arch}_seed_{seed}_{contam} (as written
by exp4_iaf.py, exp4_vae.py and exp4_gmmn.py) and writes:

    results/exp4_results.json        every *_metrics.json, as {arch: {contam: {seed: {model: {metric: value}}}}}
    results/exp4_table2.{txt,tex}    Table 2: test MMD^2 (x 10^-3) and FID, mean +/- std over seeds
    results/exp4_table5.{txt,tex}    Table 5: test MMD^2 (x 10^-3), mean +/- standard error over seeds

Both tables and the proxy AUROC per architecture and contaminant (mean +/- std
over seeds) are also printed.

Usage:
    python experiments/exp4_collect_results.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

_RUN_DIR_RE = re.compile(r"^(?P<arch>[a-z0-9_.]+)_seed_(?P<seed>\d+)_(?P<contam>kmnist|omniglot)$")

CONTAMS = [("kmnist", "K-MNIST"), ("omniglot", "Omniglot")]
ARCHS = [("iaf", "IAF", ""), ("vae", "VAE", "vae_"), ("gmmn", "GMMN", "gmmn_")]   # (key, label, model prefix)

# Table 2: the baselines of each architecture, then naive MMD and CC-MMD.
TABLE2_METHODS = {
    "iaf":  [("SWD", "swd_contam"), ("MMD", "mmd_contam"), ("CC-MMD", "ccmmd_contam")],
    "vae":  [("MSE", "vae_mse_contam"), ("Huber", "vae_huber_contam"),
             ("MMD", "vae_mmd_contam"), ("CC-MMD", "vae_ccmmd_contam")],
    "gmmn": [("GAN", "gmmn_gan_contam"), ("MMD", "gmmn_mmd_contam"), ("CC-MMD", "gmmn_ccmmd_contam")],
}

# Table 5: naive MMD, CC-MMD with I_tilde = 1, CC-MMD with the proxy, and the two classifier baselines.
TABLE5_METHODS = [
    ("Naive MMD", "mmd_contam"),
    ("CC-MMD (I~=1)", "ccmmd_i1_contam"),
    ("CC-MMD (proxy)", "ccmmd_contam"),
    ("LR prob.-weighted", "proxy_weighted_contam"),
    ("LR hard-threshold", "proxy_filtered_contam"),
]


def collect(results_dir: Path) -> dict:
    """Read every {run_dir}/*_metrics.json under results_dir into a nested dict."""
    results: dict = {}
    for run_dir in sorted(results_dir.iterdir()):
        m = _RUN_DIR_RE.match(run_dir.name) if run_dir.is_dir() else None
        if m is None:
            continue
        seed_data = results.setdefault(m["arch"], {}).setdefault(m["contam"], {}).setdefault(m["seed"], {})
        for mf in sorted(run_dir.glob("*_metrics.json")):
            with open(mf) as f:
                seed_data[mf.stem.removesuffix("_metrics")] = json.load(f)
    return results


def _values(data: dict, arch: str, contam: str, model: str, metric: str) -> np.ndarray:
    seeds = data.get(arch, {}).get(contam, {})
    return np.array([s[model][metric] for s in seeds.values() if model in s and metric in s[model]], dtype=float)


def _fmt(v: np.ndarray, spread: float, tex: bool = False) -> str:
    if not len(v):
        return "--"
    return rf"${v.mean():.1f} \pm {spread:.1f}$" if tex else f"{v.mean():.1f} +- {spread:.1f}"


def _tex_label(label: str) -> str:
    return (label.replace("CC-MMD", "CC--MMD").replace("Naive MMD", "Na\\\"ive MMD")
            .replace("I~=1", r"$\tilde{I} \equiv 1$"))


def table2(data: dict) -> tuple[list, list]:
    """Table 2 as plain-text lines and LaTeX lines."""
    header = f"{'':6s} {'Method':8s}" + "".join(f"  {lbl + ' MMD^2':>18s}  {lbl + ' FID':>18s}" for _, lbl in CONTAMS)
    txt = ["Table 2: test MMD^2 (x 10^-3) and FID, mean +- std over seeds", header, "-" * len(header)]
    tex = [r"\begin{tabular}{ll" + "cc" * len(CONTAMS) + "}", r"  \toprule",
           "  & & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{lbl}}}" for _, lbl in CONTAMS) + r" \\",
           "  & Method" + r" & MMD$^2$ ($\times 10^{-3}$) & FID" * len(CONTAMS) + r" \\"]
    for arch, arch_label, _ in ARCHS:
        tex.append(r"  \midrule")
        for i, (label, model) in enumerate(TABLE2_METHODS[arch]):
            stats = []
            for contam, _ in CONTAMS:
                mmd = _values(data, arch, contam, model, "test_mmd2") * 1e3
                fid = _values(data, arch, contam, model, "fid")
                stats += [(mmd, mmd.std() if len(mmd) else 0), (fid, fid.std() if len(fid) else 0)]
            txt.append(f"{arch_label:6s} {label:8s}" + "".join(f"  {_fmt(v, s):>18s}" for v, s in stats))
            row = [arch_label if i == 0 else "", _tex_label(label)] + [_fmt(v, s, tex=True) for v, s in stats]
            tex.append("  " + " & ".join(row) + r" \\")
    tex += [r"  \bottomrule", r"\end{tabular}"]
    return txt, tex


def table5(data: dict) -> tuple[list, list]:
    """Table 5 as plain-text lines and LaTeX lines."""
    header = f"{'':10s} {'':5s}" + "".join(f"  {label:>18s}" for label, _ in TABLE5_METHODS)
    txt = ["Table 5: test MMD^2 (x 10^-3), mean +- standard error over seeds", header, "-" * len(header)]
    tex = [r"\begin{tabular}{ll" + "c" * len(TABLE5_METHODS) + "}", r"  \toprule",
           "  & & " + " & ".join(_tex_label(label) for label, _ in TABLE5_METHODS) + r" \\"]
    for contam, contam_label in CONTAMS:
        tex.append(r"  \midrule")
        for i, (arch, arch_label, prefix) in enumerate(ARCHS):
            stats = []
            for _, model in TABLE5_METHODS:
                v = _values(data, arch, contam, prefix + model, "test_mmd2") * 1e3
                stats.append((v, v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0))
            txt.append(f"{contam_label:10s} {arch_label:5s}" + "".join(f"  {_fmt(v, s):>18s}" for v, s in stats))
            row = [contam_label if i == 0 else "", arch_label] + [_fmt(v, s, tex=True) for v, s in stats]
            tex.append("  " + " & ".join(row) + r" \\")
    tex += [r"  \bottomrule", r"\end{tabular}"]
    return txt, tex


def print_proxy_auroc(data: dict) -> None:
    print("\nProxy AUROC on the pool, mean +- std over seeds")
    for arch in sorted(data):
        for contam, contam_label in CONTAMS:
            v = _values(data, arch, contam, "proxy", "proxy_auroc")
            if len(v):
                spread = v.std(ddof=1) if len(v) > 1 else 0.0
                print(f"  {arch:12s} {contam_label:9s} {v.mean():.3f} +- {spread:.3f}  ({len(v)} seeds)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--output", type=Path, default=Path("results") / "exp4_results.json")
    args = p.parse_args()

    if not args.results_dir.exists():
        print(f"Results directory not found: {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    data = collect(args.results_dir)
    if not data:
        print("No run directories found.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {args.output}")
    for arch, contam_dict in sorted(data.items()):
        for contam, seed_dict in sorted(contam_dict.items()):
            print(f"  {arch}/{contam}: seeds {sorted(seed_dict, key=int)}")

    for name, (txt, tex) in [("table2", table2(data)), ("table5", table5(data))]:
        print("\n" + "\n".join(txt))
        for ext, lines in (("txt", txt), ("tex", tex)):
            path = args.output.parent / f"exp4_{name}.{ext}"
            path.write_text("\n".join(lines) + "\n")
            print(f"Saved {path}")
    print_proxy_auroc(data)


if __name__ == "__main__":
    main()
