"""Grid-search tuner for Random Forest.

Mirrors tune_xgboost.py and tune_xrfm.py: for each dataset, sweep a small
hyperparameter grid, score each config on the validation set, and dump the
winner to tuned_params/random_forest/<dataset>.json. RandomForestAlgorithm
can be extended to load that file when it exists, falling back to the
hardcoded defaults otherwise.

Classification uses AUC-ROC; regression uses RMSE.
"""

import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, root_mean_squared_error

# Add parent directory to path to allow imports from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processor import DATASET_CONFIG, get_prepared_data


np.random.seed(42)

GRID = {
    "n_estimators": [50, 100, 200, 300],
    "max_depth": [5, 10, 15, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

TUNED_PARAMS_DIR = Path("tuned_params/random_forest")
TUNED_PARAMS_DIR.mkdir(parents=True, exist_ok=True)


def build_model(task_type, params):
    if task_type == "classification":
        return RandomForestClassifier(
            **params,
            random_state=42,
            n_jobs=-1,
        )
    return RandomForestRegressor(
        **params,
        random_state=42,
        n_jobs=-1,
    )


for dataset_name, config in DATASET_CONFIG.items():
    out_path = TUNED_PARAMS_DIR / f"{dataset_name}.json"
    if out_path.exists():
        print(f"Skipping {dataset_name} (already tuned)")
        continue

    print(f"\n=== Tuning {dataset_name} ===")
    X_train, X_val, X_test, y_train, y_val, y_test = get_prepared_data(dataset_name)
    task_type = config["task"]

    results = []
    for combo in product(*GRID.values()):
        params = dict(zip(GRID.keys(), combo))
        model = build_model(task_type, params)
        model.fit(X_train, y_train)

        if task_type == "classification":
            probs = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, probs)
        else:
            preds = model.predict(X_val)
            score = root_mean_squared_error(y_val, preds)

        results.append({"params": params, "score": float(score)})
        print(f"  {params} -> {score:.4f}")

    if task_type == "classification":
        winner = max(results, key=lambda r: r["score"])
    else:
        winner = min(results, key=lambda r: r["score"])

    with open(out_path, "w") as f:
        json.dump(winner, f, indent=2)
    print(f"  -> winner: {winner['params']} (score={winner['score']:.4f})")
