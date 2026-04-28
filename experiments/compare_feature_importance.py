"""Compare AGOP against PCA, mutual information, and permutation importance.

Runs on the Insurance regression dataset (Regression_Mixed) — small,
interpretable, and the dataset for which we already have a full d x d AGOP
heatmap. Produces a ranking table the writeup's §2b uses to discuss agreement
on the textbook-dominant feature (smoker_yes) versus disagreement on
second-tier features.

Output:
    results/feature_importance_comparison_Regression_Mixed.csv

Invocation:
    uv run python -m experiments.compare_feature_importance
    uv run python -m experiments.compare_feature_importance --top 10
    uv run python -m experiments.compare_feature_importance --dataset Regression_Mixed
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import r2_score

from src.data_processor import DATASET_CONFIG, get_feature_names, get_prepared_data
from src.models.xrfm import xRFMAlgorithm


AGOPS_DIR = Path("results/agops")
RESULTS_DIR = Path("results")


def short(name):
    return name.replace("num__", "").replace("cat__", "")


def load_agop_diagonal(dataset_name):
    path = AGOPS_DIR / f"{dataset_name}.pt"
    data = torch.load(path, weights_only=False)
    A = data["agops"][0]
    A_np = A.cpu().numpy() if hasattr(A, "cpu") else np.asarray(A)
    return A_np if A_np.ndim == 1 else A_np.diagonal()


def pca_importance(X_train):
    """Per-feature contribution to total explained variance.

    importance_j = sum_k w_k * |V[k, j]|, where V is PCA components_ and
    w_k is the explained variance ratio of component k. Standard fudge for
    turning PCA (a method that doesn't naturally rank features) into a
    feature-importance score; flag this caveat in the writeup.
    """
    pca = PCA(n_components=min(X_train.shape) - 1, random_state=42)
    pca.fit(X_train)
    return (np.abs(pca.components_) * pca.explained_variance_ratio_[:, None]).sum(axis=0)


def permutation_importance_xrfm(model, X_test, y_test, n_repeats=10, seed=42):
    """Drop in feature j -> shuffle column j -> measure score drop.

    Score is R^2 since Insurance is regression. Larger drop = more important.
    """
    rng = np.random.default_rng(seed)
    baseline = r2_score(y_test, model.predict(X_test))
    d = X_test.shape[1]
    importances = np.zeros(d)
    for j in range(d):
        drops = np.zeros(n_repeats)
        for r in range(n_repeats):
            X_perm = X_test.copy()
            rng.shuffle(X_perm[:, j])
            drops[r] = baseline - r2_score(y_test, model.predict(X_perm))
        importances[j] = drops.mean()
    return importances


def rank(values):
    """Dense ranks where rank 1 = largest value."""
    order = np.argsort(-values)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def build_comparison_table(feature_names, importances):
    """importances: dict[method_name -> np.ndarray of length d]."""
    rows = {"feature": [short(n) for n in feature_names]}
    for method, vals in importances.items():
        rows[f"{method}_importance"] = vals
        rows[f"{method}_rank"] = rank(vals)
    return pd.DataFrame(rows)


def print_top_k_table(df, top_k):
    methods = ["AGOP", "PCA", "MI", "Permutation"]
    cols = {m: df.sort_values(f"{m}_rank").head(top_k)["feature"].tolist() for m in methods}
    print(f"\nTop-{top_k} features by method:")
    header = "  rank  " + "  ".join(f"{m:<25}" for m in methods)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i in range(top_k):
        row = f"  {i+1:<4}  " + "  ".join(f"{cols[m][i]:<25}" for m in methods)
        print(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Regression_Mixed",
                        help="Dataset stem (default: Regression_Mixed = Insurance).")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--n-repeats", type=int, default=10,
                        help="Permutation importance repeats.")
    args = parser.parse_args()

    if args.dataset not in DATASET_CONFIG:
        raise SystemExit(f"Unknown dataset: {args.dataset}")
    if DATASET_CONFIG[args.dataset]["task"] != "regression":
        raise SystemExit(f"{args.dataset} is not regression; permutation R^2 won't apply.")

    print(f"=== Feature-importance comparison: {args.dataset} ===")

    print("Loading data...")
    X_tr, X_va, X_te, y_tr, y_va, y_te = get_prepared_data(args.dataset)
    feature_names = get_feature_names(args.dataset)
    print(f"  X_train: {X_tr.shape}  |  d = {len(feature_names)}")

    print("Loading AGOP diagonal...")
    agop_imp = load_agop_diagonal(args.dataset)

    print("Computing PCA importance...")
    pca_imp = pca_importance(X_tr)

    print("Computing mutual information...")
    mi_imp = mutual_info_regression(X_tr, y_tr, random_state=42)

    print(f"Training xRFM for permutation importance (n_repeats={args.n_repeats})...")
    model = xRFMAlgorithm(dataset_name=args.dataset, task_type="regression")
    model.train(X_tr, y_tr, X_va, y_va)
    perm_imp = permutation_importance_xrfm(model, X_te, y_te, n_repeats=args.n_repeats)

    df = build_comparison_table(
        feature_names,
        {"AGOP": agop_imp, "PCA": pca_imp, "MI": mi_imp, "Permutation": perm_imp},
    )

    print_top_k_table(df, args.top)

    out_path = RESULTS_DIR / f"feature_importance_comparison_{args.dataset}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nFull comparison saved -> {out_path}")

    print("\nAgreement check (top-1 across all methods):")
    for m in ("AGOP", "PCA", "MI", "Permutation"):
        top1 = df.sort_values(f"{m}_rank").iloc[0]["feature"]
        print(f"  {m:<12} top-1: {top1}")


if __name__ == "__main__":
    main()
