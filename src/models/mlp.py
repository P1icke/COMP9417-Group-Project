import json
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import precision_recall_curve
from imblearn.pipeline import Pipeline as IMLPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from src.models.base_model import BaseModel


class MLPAlgorithm(BaseModel):
    IMBALANCED_DATASETS = {"Classification_n_gt_10k", "Classification_d_gt_50"}

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

        if self.task_type == "classification":
            mlp = MLPClassifier(**params, max_iter=2000, early_stopping=True, random_state=42)
            if dataset_name in self.IMBALANCED_DATASETS:
                # 10:1 undersample then 2:1 SMOTE — matches mlp_cv.ipynb's strategy
                # for highly imbalanced classification.
                self.model = IMLPipeline([
                    ('under', RandomUnderSampler(sampling_strategy=0.2, random_state=42)),
                    ('over', SMOTE(sampling_strategy=0.5, random_state=42)),
                    ('mlp', mlp),
                ])
                self._tune_threshold = True
            else:
                self.model = mlp
        else:
            self.model = MLPRegressor(**params, max_iter=2000, early_stopping=True, random_state=42)

    def train(self, X_train, y_train, X_val, y_val):
        self.model.fit(X_train, y_train)

        if self._tune_threshold:
            val_prob = self.model.predict_proba(X_val)[:, 1]
            precision, recall, thresholds = precision_recall_curve(y_val, val_prob)
            f1 = 2 * precision * recall / (precision + recall + 1e-12)
            best_idx = int(np.argmax(f1[:-1]))
            self.threshold = thresholds[best_idx]

    def predict(self, X_test):
        if self._tune_threshold:
            prob = self.model.predict_proba(X_test)[:, 1]
            return (prob >= self.threshold).astype(int)
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)
