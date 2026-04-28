import numpy as np
import torch

# Disable CUDA to avoid GPU compatibility issues
torch.cuda.is_available = lambda: True

from src.models.base_model import BaseModel
from xrfm import xRFM
from pathlib import Path
import json

def _as_writable(arr):
    # pandas .values can yield a read-only view; torch segfaults when it writes to one.
    # np.ascontiguousarray is a no-op if already contiguous and preserves the read-only flag,
    # so we force a real copy here.
    return np.array(arr, copy=True)


class xRFMAlgorithm(BaseModel):
    def __init__(self, dataset_name, task_type):
        super().__init__(dataset_name, task_type)

        tuning_metric = "mse" if task_type == "regression" else "accuracy"

        tuned_path = f"tuned_params/xrfm/{dataset_name}.json"
        # check if the params have been tuned


        if Path(tuned_path).exists():
            with open(tuned_path) as f:
                tuned = json.load(f)
            tuned_params = tuned["params"]

            default_rfm_params = {
                'model': {
                    "kernel": "l2_high_dim",
                    "exponent": 1.0,
                    "bandwidth": tuned_params["bandwidth"],
                    "diag": tuned_params["diag"],
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
            self.model = xRFM(
                tuning_metric=tuning_metric,
                random_state=42,
                default_rfm_params=default_rfm_params,
                n_threads=1,
            )
        else:
        # n_threads=1 avoids an OpenMP conflict on macOS when xgboost + sklearn + xrfm
        # are loaded in the same process (causes SIGSEGV during RFM fit otherwise).
            self.model = xRFM(
                tuning_metric=tuning_metric,
                random_state=42,
                n_threads=1,
            )

    def train(self, X_train, y_train, X_val, y_val):
        self.model.fit(
            _as_writable(X_train),
            _as_writable(y_train),
            _as_writable(X_val),
            _as_writable(y_val),
        )

    def predict(self, X_test):
        return self.model.predict(_as_writable(X_test))

    def predict_proba(self, X_test):
        # xRFM has a predict proba method
        # so we want the output as a probably, instead of a yes/no answer
        return self.model.predict_proba(_as_writable(X_test))

    def get_leaf_agops(self):
        return self.model.collect_best_agops()
