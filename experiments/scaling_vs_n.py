"""Scaling-vs-n sweep on one large dataset (n > 10,000).

Subsamples the training set at several sizes, retrains each of the four models
on the subsample (validation and test sets stay full), and records test
performance + training time. Produces a two-panel plot:

  - left panel:  test metric (RMSE for regression, Accuracy for classification) vs n
  - right panel: training time (s, log-y) vs n

Spec requirement (Project §4 Results, third bullet): "On one large dataset
(n > 10,000), subsample the training set at several sizes and plot test
performance and training time versus n for all models."

Defaults to Classification_n_gt_10k (Online Shoppers) — the full sweep
finishes in a few minutes. Pass --dataset Regression_n_gt_10k for the CASP
version, which is dramatic but slow (xRFM dominates the runtime; budget
30-60 min).

Outputs (long-form so the same CSV drives the plot and any later analysis):
    results/scaling_vs_n_<dataset>.csv
    results/scaling_vs_n_<dataset>.png

Invocation:
    uv run python -m experiments.scaling_vs_n
    uv run python -m experiments.scaling_vs_n --dataset Regression_n_gt_10k
    uv run python -m experiments.scaling_vs_n --sizes 500 1000 2000 4000 8000

The CSV is *overwritten* on each run so partial-rerun edge cases don't
silently double-count points; if you want to keep an old sweep, copy it
aside first.
"""
import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless safe (CI / SSH)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

from src.data_processor import DATASET_CONFIG, get_prepared_data
from src.models.mlp import MLPAlgorithm
from src.models.random_forest import RandomForestAlgorithm
from src.models.xgboost import XGBoostAlgorithm
from src.models.xrfm import xRFMAlgorithm


SEED = 42

# Dataset-specific default sweeps. Endpoint is appended automatically if it
# isn't already covered.
DATASET_DEFAULTS = {
    "Classification_n_gt_10k": [500, 1000, 2000, 4000, 8000],
    "Regression_n_gt_10k": [1000, 2500, 5000, 10000, 20000],
}
DEFAULT_DATASET = "Classification_n_gt_10k"

MODEL_CTORS = {
    "xRFM": xRFMAlgorithm,
    "XGBoost": XGBoostAlgorithm,
    "RandomForest": RandomForestAlgorithm,
    "MLP": MLPAlgorithm,
}

OUT_DIR = Path("results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        choices=list(DATASET_DEFAULTS),
        help="Which n>10k dataset to sweep (default: %(default)s).",
    )
    p.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=None,
        help="Subsample sizes (defaults are dataset-specific).",
    )
    return p.parse_args()


def subsample_train(X, y, n, task):
    """Return a length-n training subset.

    For classification we stratify on y so small samples still see both
    classes — important on imbalanced datasets like Online Shoppers
    (~15.5% positive). For regression we just shuffle and slice.
    """
    if n >= len(X):
        return X, y
    if task == "classification":
        X_keep, _, y_keep, _ = train_test_split(
            X, y, train_size=n, stratify=y, random_state=SEED
        )
        return X_keep, y_keep
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(X))[:n]
    return X[idx], y[idx]


def score(task, y_true, predictions, probas):
    """Headline metric first (used for plotting); AUC tagged on for classification."""
    if task == "regression":
        return {"RMSE": root_mean_squared_error(y_true, predictions)}
    out = {"Accuracy": accuracy_score(y_true, predictions)}
    if probas is not None:
        try:
            out["AUC-ROC"] = roc_auc_score(y_true, probas)
        except Exception:
            pass
    return out


def run_one(model_name, ctor, dataset_name, task, X_tr, y_tr, X_va, y_va, X_te, y_te):
    """Train a fresh model on the (already subsampled) train set; evaluate on full test."""
    model = ctor(dataset_name=dataset_name, task_type=task)

    t0 = time.time()
    model.train(X_tr, y_tr, X_va, y_va)
    train_time = time.time() - t0

    t0 = time.time()
    predictions = model.predict(X_te)
    predict_time = time.time() - t0

    probas = None
    if task == "classification" and hasattr(model, "predict_proba"):
        try:
            probas = model.predict_proba(X_te)[:, 1]
        except Exception as e:
            print(f"      AUC unavailable for {model_name}: {e}")

    metrics = score(task, y_te, predictions, probas)
    return train_time, predict_time / max(len(X_te), 1), metrics


def plot_results(df, dataset_name, task, out_path):
    metric = "RMSE" if task == "regression" else "Accuracy"

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    metric_lines = df.pivot_table(index="n", columns="model", values=metric)
    time_lines = df.pivot_table(index="n", columns="model", values="train_time")

    # Lock model order so colours/legend match across panels.
    model_order = [m for m in MODEL_CTORS if m in metric_lines.columns]

    for col in model_order:
        axes[0].plot(metric_lines.index, metric_lines[col], marker="o", label=col)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Training set size (n)")
    axes[0].set_ylabel(metric)
    axes[0].set_title(f"Test {metric} vs n")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    for col in model_order:
        axes[1].plot(time_lines.index, time_lines[col], marker="o", label=col)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Training set size (n)")
    axes[1].set_ylabel("Training time (s)")
    axes[1].set_title("Training time vs n")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()

    fig.suptitle(f"Scaling behaviour on {dataset_name}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved figure -> {out_path}")


def main():
    args = parse_args()
    dataset_name = args.dataset
    task = DATASET_CONFIG[dataset_name]["task"]
    requested_sizes = args.sizes if args.sizes else DATASET_DEFAULTS[dataset_name]

    print(f"=== Scaling sweep on {dataset_name} (task={task}) ===")

    X_tr_full, X_va, X_te, y_tr_full, y_va, y_te = get_prepared_data(dataset_name)
    full_n = len(X_tr_full)
    print(f"Full train n = {full_n}, val n = {len(X_va)}, test n = {len(X_te)}")

    sweep_sizes = sorted({s for s in requested_sizes if s <= full_n})
    if not sweep_sizes or sweep_sizes[-1] < full_n:
        sweep_sizes.append(full_n)
    print(f"Subsample sizes: {sweep_sizes}\n")

    csv_path = OUT_DIR / f"scaling_vs_n_{dataset_name}.csv"
    fig_path = OUT_DIR / f"scaling_vs_n_{dataset_name}.png"

    # Fresh CSV each run — see module docstring.
    if csv_path.exists():
        csv_path.unlink()

    rows = []
    for n in sweep_sizes:
        X_tr, y_tr = subsample_train(X_tr_full, y_tr_full, n, task)
        if task == "classification":
            pos_rate = float(np.mean(y_tr))
            print(f"--- n={n} (pos rate {pos_rate:.3f}) ---")
        else:
            print(f"--- n={n} ---")

        for model_name, ctor in MODEL_CTORS.items():
            print(f"  [{model_name}]", flush=True)
            try:
                train_time, infer_time, metrics = run_one(
                    model_name, ctor, dataset_name, task,
                    X_tr, y_tr, X_va, y_va, X_te, y_te,
                )
            except Exception as e:
                print(f"      failed: {e}")
                continue

            row = {
                "dataset": dataset_name,
                "n": n,
                "model": model_name,
                "train_time": round(train_time, 4),
                "infer_time_per_sample": round(infer_time, 6),
            }
            row.update({k: round(v, 4) for k, v in metrics.items()})
            rows.append(row)
            tagline = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            print(f"      {tagline}, train {train_time:.2f}s")

            # Append-as-you-go so a mid-sweep crash keeps partial data.
            pd.DataFrame([row]).to_csv(
                csv_path,
                mode="a",
                header=not csv_path.exists(),
                index=False,
            )

    if not rows:
        print("\nNo successful runs; skipping plot.")
        return

    df = pd.DataFrame(rows)
    plot_results(df, dataset_name, task, fig_path)
    print(f"\nDone. CSV -> {csv_path}")


if __name__ == "__main__":
    main()
