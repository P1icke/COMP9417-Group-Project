import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, root_mean_squared_error
from xrfm import xRFM

from src.data_processor import get_prepared_data, DATASET_CONFIG


"""Load your data"""

GRID = [
    {"bandwidth": 1, "diag": True},
    {"bandwidth": 1, "diag": False}, 
    {"bandwidth": 5, "diag": True},
    {"bandwidth": 5, "diag": False},  
    {"bandwidth": 10, "diag": True},
    {"bandwidth": 10, "diag": False},  
    {"bandwidth": 25, "diag": True},
    {"bandwidth": 25, "diag": False}, 
    {"bandwidth": 100, "diag": True},
    {"bandwidth": 100, "diag": False},  
]

DATASETS = DATASET_CONFIG.items()

TUNED_PARAMS_DIR = Path("tuned_params/xrfm")
TUNED_PARAMS_DIR.mkdir(parents=True, exist_ok=True)


"""Try a bunch of hyperparameter combinations"""


"""Traion xRFM with that training data, and score it"""

for dataset_name, config in DATASETS:
    if (TUNED_PARAMS_DIR / f"{dataset_name}.json").exists():
        print(f"Skipping {dataset_name} (already tuned)")
        continue

    print(f"\n=== Tuning {dataset_name} ===")
    X_train, X_val, X_test, y_train, y_val, y_test = get_prepared_data(dataset_name)
    task_type = config["task"]

    results = []
    for item in GRID:
        default_rfm_params = {
            'model': {
                "kernel": "l2_high_dim",
                "exponent": 1.0,
                "bandwidth": item["bandwidth"],
                "diag": item["diag"],
                "bandwidth_mode": "constant",
            },
            'fit': {
                "get_agop_best_model": True,
                "return_best_params": False,
                "reg": 1e-3,
                "iters": 0,
                "early_stop_rfm": False,
            },
        }

        model = xRFM(
            tuning_metric="accuracy" if task_type == "classification" else "mse",   
            random_state=42,
            default_rfm_params=default_rfm_params,
            n_threads=1,
            
            )
        model.fit(X_train, y_train, X_val, y_val)
        predictions = model.predict(X_val)

        if task_type == "classification":
            score = accuracy_score(y_val, predictions)
        else:
            score = root_mean_squared_error(y_val, predictions)
  
        results.append({"params": item, "score": score})
    
    # pick a winner
    """Pick the combination with the best val score"""

    if task_type == "classification":
        winner = max(results, key=lambda r: r["score"])
    else:
        winner = min(results, key=lambda r: r["score"])

    """Save those winning settings to a file"""
    with open(TUNED_PARAMS_DIR / f"{dataset_name}.json", "w") as file:
        json.dump(winner, file, indent=2)



