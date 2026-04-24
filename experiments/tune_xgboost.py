"""Grid-search tuner for XGBoost.

Mirrors tune_xrfm.py: for each dataset, sweep a small hyperparameter grid,
score each config on the validation set, and dump the winner to
tuned_params/xgboost/<dataset>.json. XGBoostAlgorithm.__init__ loads that
file when it exists, falling back to the hardcoded defaults otherwise.

Classification uses AUC-ROC (accuracy is misleading on Classification_n_gt_10k
where the majority baseline is 99.83%); regression uses RMSE.
"""

import json
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, root_mean_squared_error
from xgboost import XGBClassifier, XGBRegressor

from src.data_processor import DATASET_CONFIG, get_prepared_data


np.random.seed(42)

GRID = {
    "max_depth": [1, 2, 3, 5, 7, 9, 11],
    "learning_rate": [0.005, 0.01, 0.05, 0.1, 0.2, 0.3],
}

# XGBoost is fast enough on Classification_n_gt_10k to tune directly — no skip
# list needed (unlike tune_xrfm.py, where the same dataset is runtime-prohibitive).
TUNED_PARAMS_DIR = Path("tuned_params/xgboost")
TUNED_PARAMS_DIR.mkdir(parents=True, exist_ok=True)


def build_model(task_type, params, y_train):
    # 4000 ceiling so low-lr configs (e.g. 0.005) have room to converge before
    # hitting the wall. Previous run showed lr=0.01 on Classification_n_gt_10k
    # still improving at iter 1999; 4000 gives that regime breathing room.
    n_estimators = 4000
    early_stopping = 20
    if task_type == "classification":
        pos = int((y_train == 1).sum())
        neg = int((y_train == 0).sum())
        spw = neg / pos if pos > 0 else 1.0
        return XGBClassifier(
            **params,
            n_estimators=n_estimators,
            early_stopping_rounds=early_stopping,
            random_state=42,
            eval_metric="logloss",
            scale_pos_weight=spw,
        )
    return XGBRegressor(
        **params,
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping,
        random_state=42,
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
        model = build_model(task_type, params, y_train)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        if task_type == "classification":
            probs = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, probs)
        else:
            preds = model.predict(X_val)
            score = root_mean_squared_error(y_val, preds)

        # Bake the early-stopping result into n_estimators so the wrapper can
        # reload without needing the val set to rediscover it.
        tuned = dict(params)
        tuned["n_estimators"] = int(model.best_iteration) + 1
        results.append({"params": tuned, "score": float(score)})
        print(f"  {params} -> {score:.4f} (best_iter={model.best_iteration})")

    if task_type == "classification":
        winner = max(results, key=lambda r: r["score"])
    else:
        winner = min(results, key=lambda r: r["score"])

    with open(out_path, "w") as f:
        json.dump(winner, f, indent=2)
    print(f"  -> winner: {winner['params']} (score={winner['score']:.4f})")
