import json
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_curve
from experiments.tune_mlp import build_search_pipeline, is_imbalanced
from src.data_processor import get_prepared_data
from src.models.base_model import BaseModel


class MLPAlgorithm(BaseModel):
    def __init__(self, dataset_name, task_type):
        super().__init__(dataset_name, task_type)
        self.hyperparameters = {
            "Regression_n_gt_10k":      {
                'activation': 'tanh',
                'alpha': np.float64(0.08591342830048515),
                'hidden_layer_sizes': (128, 64),
                'learning_rate_init': np.float64(0.00615533906228275),
            },
            "Regression_d_gt_50":       {
                'activation': 'relu',
                'alpha': np.float64(0.07423121900844358),
                'hidden_layer_sizes': (64, 32),
                'learning_rate_init': np.float64(0.006741738304421186),
            },
            "Regression_Mixed":         {
                'activation': 'relu',
                'alpha': np.float64(0.0066157058689567585),
                'hidden_layer_sizes': (128, 64),
                'learning_rate_init': np.float64(0.009776977915629067),
            },
            "Classification_n_gt_10k":  {
                'activation': 'tanh',
                'alpha': np.float64(2.1057814970278994e-05),
                'hidden_layer_sizes': (128, 64),
                'learning_rate_init': np.float64(0.00029872741995638395),
            },
            "Classification_d_gt_50":   {
                'activation': 'tanh',
                'alpha': np.float64(0.008111253665497063),
                'hidden_layer_sizes': (64, 32),
                'learning_rate_init': np.float64(0.00015030900645056822),
            },
            "Classification_Mixed":     {
                'activation': 'relu',
                'alpha': np.float64(2.6422690597255385e-05),
                'hidden_layer_sizes': (128, 64),
                'learning_rate_init': np.float64(0.004048966222584676),
            },
        }

        tuned_threshold = None
        tuned_path = Path(f"tuned_params/mlp/{dataset_name}.json")
        if tuned_path.exists():
            with open(tuned_path) as f:
                record = json.load(f)
            params = dict(record.get("params", {}))
            if "hidden_layer_sizes" in params and isinstance(params["hidden_layer_sizes"], list):
                params["hidden_layer_sizes"] = tuple(params["hidden_layer_sizes"])
            tuned_threshold = record.get("threshold")
        else:
            params = self.hyperparameters.get(dataset_name, {})

        self._tune_threshold = False
        self.threshold = tuned_threshold if tuned_threshold is not None else 0.5

        X_train, X_val, X_test, y_train, y_val, y_test = get_prepared_data(dataset_name)
        self.model, label = build_search_pipeline(task_type, X_train, y_train, params, preprocess=False)

        if is_imbalanced(label):
            self._tune_threshold = True

    def train(self, X_train, y_train, X_val, y_val):
        self.model.fit(X_train, y_train)

        if self._tune_threshold:
            val_prob = self.model.predict_proba(X_val)[:, 1]
            precision, recall, thresholds = precision_recall_curve(y_val, val_prob)
            f1 = 2 * precision * recall / (precision + recall + 1e-12)
            best_idx = int(np.argmax(f1[:-1]))
            self.threshold = thresholds[best_idx]

    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        if self.task_type == "classification":
            return self.model.predict_proba(X_test)
        raise NotImplementedError("predict_proba is only available for classification tasks")
