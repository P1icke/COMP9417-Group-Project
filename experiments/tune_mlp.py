"""Randomized-search tuner for scikit-learn MLP.

Mirrors the methodology from mlp_cv.ipynb: RandomizedSearchCV over
hidden_layer_sizes / alpha / learning_rate_init / activation, with imbalanced
classification handled via undersample-then-SMOTE inside an imblearn Pipeline
and a post-hoc F1-optimised threshold tuned on the validation split.

Per-dataset winners are written to tuned_params/mlp/<dataset>.json and the
console trace is mirrored to results/tuning_mlp.txt. MLPAlgorithm picks up the
JSON automatically when present.
"""

import json
import sys
import time
from enum import Enum
from pathlib import Path
from pprint import pformat

import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as IMLPipeline
from imblearn.under_sampling import RandomUnderSampler

from src.data_processor import DATASET_CONFIG, _build_preprocessor, get_prepared_data


np.random.seed(42)

TUNED_PARAMS_DIR = Path("tuned_params/mlp")
TUNED_PARAMS_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = Path("results/tuning_mlp.txt")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


class _Tee:
    """Mirror writes to stdout and the tuning log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


_log_file = open(LOG_PATH, "a", buffering=1)
sys.stdout = _Tee(sys.__stdout__, _log_file)


class ImbalanceRanking(Enum):
    BALANCED = 0.4
    MILD_IMBALANCE = 0.2
    MODERATE_IMBALANCE = 0.05
    SEVERE_IMBALANCE = 0.01


class ImbalanceLabels(Enum):
    BALANCED = "balanced"
    MILD_IMBALANCE = "mild_imbalanced"
    MODERATE_IMBALANCE = "moderate_imbalance"
    SEVERE_IMBALANCE = "severe_imbalance"
    EXTREME_IMBALANCE = "extreme_imbalance"


param_dist = {
    "mlp__hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)],
    "mlp__alpha": loguniform(1e-5, 1e-1),
    "mlp__learning_rate_init": loguniform(1e-4, 1e-2),
    "mlp__activation": ["relu", "tanh"],
}


def class_distribution(y) -> dict:
    classes, counts = np.unique(y, return_counts=True)
    n = counts.sum()
    proportions = counts / n
    return {
        "n_classes": len(classes),
        "counts": dict(zip(classes.tolist(), counts.tolist())),
        "proportions": dict(zip(classes.tolist(), proportions.tolist())),
        "minority_proportion": float(proportions.min()),
        "imbalance_ratio": float(counts.max() / counts.min()),
    }


def imbalance_severity(y) -> ImbalanceLabels:
    p = class_distribution(y)["minority_proportion"]
    if p >= ImbalanceRanking.BALANCED.value:
        return ImbalanceLabels.BALANCED
    if p >= ImbalanceRanking.MILD_IMBALANCE.value:
        return ImbalanceLabels.MILD_IMBALANCE
    if p >= ImbalanceRanking.MODERATE_IMBALANCE.value:
        return ImbalanceLabels.MODERATE_IMBALANCE
    if p >= ImbalanceRanking.SEVERE_IMBALANCE.value:
        return ImbalanceLabels.SEVERE_IMBALANCE
    return ImbalanceLabels.EXTREME_IMBALANCE


def is_imbalanced(label: ImbalanceLabels) -> bool:
    return label in {
        ImbalanceLabels.MODERATE_IMBALANCE,
        ImbalanceLabels.SEVERE_IMBALANCE,
        ImbalanceLabels.EXTREME_IMBALANCE,
    }


def build_search_pipeline(task_type: str, X_train, y_train, params=None, preprocess=True):
    """Pipeline used during RandomizedSearchCV — params are injected by the
    search, so the estimator is constructed without them."""
    if params is None:
        params = {}
    preprocessor = _build_preprocessor(X_train) if preprocess else None

    if task_type == "classification":
        label = imbalance_severity(y_train)
        mlp = MLPClassifier(**params, max_iter=2000, early_stopping=True, random_state=42)

        if label == ImbalanceLabels.MODERATE_IMBALANCE:
            return IMLPipeline([
                ("preprocess", preprocessor),
                ("over", SMOTE(sampling_strategy=0.5, random_state=42)),
                ("mlp", mlp),
            ]), label
        if label in {ImbalanceLabels.SEVERE_IMBALANCE, ImbalanceLabels.EXTREME_IMBALANCE}:
            return IMLPipeline([
                ("preprocess", preprocessor),
                ("under", RandomUnderSampler(sampling_strategy=0.1, random_state=42)),
                ("over", SMOTE(sampling_strategy=0.5, random_state=42)),
                ("mlp", mlp),
            ]), label
        return Pipeline([("preprocess", preprocessor), ("mlp", mlp)]), label

    return Pipeline([
        ("preprocess", preprocessor),
        ("mlp", MLPRegressor(**params, max_iter=2000, early_stopping=True, random_state=42)),
    ]), None


def strip_prefix(params: dict) -> dict:
    return {k.removeprefix("mlp__"): v for k, v in params.items()}


def to_jsonable(value):
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def tune_dataset(dataset_name: str, config: dict) -> None:
    out_path = TUNED_PARAMS_DIR / f"{dataset_name}.json"
    if out_path.exists():
        print(f"Skipping {dataset_name} (already tuned)")
        return

    print(f"\n=== Tuning {dataset_name} ===")
    print("Loading & processing dataset...")
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = get_prepared_data(
            dataset_name, return_raw=True
        )
    except Exception as e:
        print(f"Skipping {dataset_name} due to error: {e}")
        return

    task_type = config["task"]
    pipe, label = build_search_pipeline(task_type, X_train, y_train)

    if task_type == "classification":
        scoring = "roc_auc"
        cv = StratifiedKFold(5, shuffle=True, random_state=42)
        imbalanced = is_imbalanced(label)
        if imbalanced:
            # Hold validation out so threshold tuning has an unseen split.
            X_search, y_search = X_train, y_train
        else:
            X_search = pd.concat([X_train, X_val], axis=0)
            y_search = np.concatenate([y_train, y_val], axis=0)
    else:
        scoring = "r2"
        cv = KFold(5, shuffle=True, random_state=42)
        imbalanced = False
        X_search = pd.concat([X_train, X_val], axis=0)
        y_search = np.concatenate([y_train, y_val], axis=0)

    if task_type == "classification":
        print(f"Imbalance: {label.value}")

    search = RandomizedSearchCV(
        pipe,
        param_dist,
        n_iter=60,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        random_state=(hash(dataset_name) % (2**32)),
    )

    print(f"Fitting RandomizedSearchCV (n_iter=60, scoring={scoring})...")
    t0 = time.time()
    search.fit(X_search, y_search)
    elapsed = time.time() - t0

    best_params = strip_prefix(search.best_params_)
    print(f"Search finished in {elapsed:.1f}s")
    print("Best parameters:")
    print(pformat(best_params))
    print(f"Best CV score ({scoring}): {search.best_score_:.4f}")

    record = {
        "params": to_jsonable(best_params),
        "cv_score": float(search.best_score_),
        "scoring": scoring,
    }

    if task_type == "classification" and imbalanced:
        print("Tuning classification threshold on validation set...")
        best_pipe = search.best_estimator_
        val_prob = best_pipe.predict_proba(X_val)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_val, val_prob)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        best_idx = int(np.argmax(f1[:-1]))
        best_threshold = float(thresholds[best_idx])
        print(f"Best threshold: {best_threshold:.6f}")
        print(f"Validation F1:        {f1[best_idx]:.4f}")
        print(f"Validation Precision: {precision[best_idx]:.4f}")
        print(f"Validation Recall:    {recall[best_idx]:.4f}")

        test_prob = best_pipe.predict_proba(X_test)[:, 1]
        test_pred = (test_prob >= best_threshold).astype(int)
        print(f"Test AUC-ROC: {roc_auc_score(y_test, test_prob):.4f}")
        print(f"Test F1 @ tuned threshold: {f1_score(y_test, test_pred):.4f}")

        record["threshold"] = best_threshold

    elif task_type == "classification":
        best_pipe = search.best_estimator_
        test_prob = best_pipe.predict_proba(X_test)[:, 1]
        test_pred = best_pipe.predict(X_test)
        print(f"Test AUC-ROC: {roc_auc_score(y_test, test_prob):.4f}")
        print(f"Test F1 @ default threshold: {f1_score(y_test, test_pred):.4f}")

    else:
        best_pipe = search.best_estimator_
        y_pred = best_pipe.predict(X_test)
        print(f"Test R^2:  {r2_score(y_test, y_pred):.4f}")
        print(f"Test RMSE: {root_mean_squared_error(y_test, y_pred):.4f}")

    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"Saved tuned params -> {out_path}")


def main():
    print(f"\n========= MLP tuning run @ {time.strftime('%Y-%m-%d %H:%M:%S')} =========")
    for dataset_name, config in DATASET_CONFIG.items():
        tune_dataset(dataset_name, config)
    print("\nDone.")


if __name__ == "__main__":
    main()
