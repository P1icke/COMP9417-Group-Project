"""Inspect and visualise saved xRFM AGOP matrices.

Loads each `results/agops/<dataset>.pt` and prints the top-K features by
diagonal importance. Optionally saves a bar chart per dataset, plus a
heatmap for datasets whose AGOP was stored as a full (d, d) matrix
(i.e. those tuned with diag=False).

Invocation:
    uv run python -m experiments.view_agops                     # all datasets, plots + text
    uv run python -m experiments.view_agops --dataset Regression_Mixed
    uv run python -m experiments.view_agops --no-plots          # text-only
    uv run python -m experiments.view_agops --top 5
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt


AGOPS_DIR = Path("results/agops")
PLOTS_DIR = AGOPS_DIR / "plots"


def load_agop(name):
    data = torch.load(AGOPS_DIR / f"{name}.pt", weights_only=False)
    A = data["agops"][0]
    A_np = A.cpu().numpy() if hasattr(A, "cpu") else np.asarray(A)
    diag = A_np if A_np.ndim == 1 else A_np.diagonal()
    return A_np, diag, data["feature_names"]


def short(name):
    return name.replace("num__", "").replace("cat__", "")


def print_summary(name, A_np, diag, feature_names, top_k):
    is_full = A_np.ndim == 2
    print(f"\n=== {name} ===")
    print(f"  storage : {'full (d, d)' if is_full else 'diagonal-only (d,)'}, d={len(diag)}")
    print(f"  diag    : min={diag.min():.3e}  max={diag.max():.3e}  mean={diag.mean():.3e}")
    order = np.argsort(-diag)
    print(f"  top-{top_k} features by AGOP diagonal:")
    for rank, i in enumerate(order[:top_k], 1):
        print(f"    {rank:2d}. {short(feature_names[i]):40s} {diag[i]:.4f}")


def plot_importance_bar(name, diag, feature_names, top_k, out_path):
    order = np.argsort(-diag)[:top_k][::-1]  # reversed so highest is at top
    fig, ax = plt.subplots(figsize=(8, max(3, 0.32 * len(order))))
    ax.barh([short(feature_names[i]) for i in order], diag[order], color="#3b6fb6")
    ax.set_xlabel("AGOP diagonal (feature importance, normalised)")
    ax.set_title(f"{name}  —  top-{top_k} features")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"    saved {out_path}")


def plot_full_heatmap(name, A_np, feature_names, out_path):
    labels = [short(n) for n in feature_names]
    fig, ax = plt.subplots(figsize=(max(6, 0.28 * len(labels)), max(5, 0.28 * len(labels))))
    im = ax.imshow(A_np, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(f"{name}  —  full AGOP matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"    saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="View saved xRFM AGOP files.")
    parser.add_argument("--dataset", help="Single dataset stem (e.g. Regression_Mixed).")
    parser.add_argument("--top", type=int, default=10, help="Top-K features to print/plot.")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation.")
    args = parser.parse_args()

    if args.dataset:
        files = [AGOPS_DIR / f"{args.dataset}.pt"]
    else:
        files = sorted(AGOPS_DIR.glob("*.pt"))

    if not args.no_plots:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    for pt in files:
        if not pt.exists():
            print(f"missing: {pt}")
            continue
        name = pt.stem
        A_np, diag, feature_names = load_agop(name)
        print_summary(name, A_np, diag, feature_names, top_k=args.top)
        if args.no_plots:
            continue
        plot_importance_bar(name, diag, feature_names, args.top, PLOTS_DIR / f"{name}_importance.png")
        if A_np.ndim == 2:
            plot_full_heatmap(name, A_np, feature_names, PLOTS_DIR / f"{name}_heatmap.png")


if __name__ == "__main__":
    main()
