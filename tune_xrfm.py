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


"""Try a bunch of hyperparameter combinations"""


"""Traion xRFM with that training data, and score it"""

for dataset_name, config in DATASETS:
    X_train, X_val, X_test, y_train, y_val, y_test = get_prepared_data(dataset_name) 
    task_type = config["task"]

    for item in GRID:
        default_rfm_params = {'model': {
            "bandwidth": item["bandwidth"],
            "diag": item["diag"]
        }}

        model = xRFM(
            tuning_metric="accuracy" if task_type == "classification" else "mse",   
            random_state=42,
            default_rfm_params=default_rfm_params,
            n_threads=1,
            
            )
        model.fit(X_train, X_val, X_test, y_train, y_val, y_test)

        model.predict(X_val)

"""Pick the combination with the best val score"""


"""Save those winning settings to a file"""