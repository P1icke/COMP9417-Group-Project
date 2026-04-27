import json
from pathlib import Path

from src.models.base_model import BaseModel
from src.data_processor import get_prepared_data, _build_preprocessor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve
from imblearn.pipeline import Pipeline as IMLPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import numpy as np

TUNED_PARAMS_DIR = Path("tuned_params/mlp")


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

        params = self.hyperparameters.get(dataset_name)

        # Override defaults with tuned_params/mlp/<dataset>.json if present —
        # tune_mlp.py writes that file with the RandomizedSearchCV winner and,
        # for imbalanced classification, the F1-optimised threshold.
        tuned_threshold = None
        tuned_path = TUNED_PARAMS_DIR / f"{dataset_name}.json"
        if tuned_path.exists():
            with open(tuned_path) as f:
                record = json.load(f)
            tuned = dict(record.get("params", {}))
            if "hidden_layer_sizes" in tuned and isinstance(tuned["hidden_layer_sizes"], list):
                tuned["hidden_layer_sizes"] = tuple(tuned["hidden_layer_sizes"])
            params = tuned
            tuned_threshold = record.get("threshold")

        # Reload raw splits so the model can own its own preprocess + resample
        # pipeline. The deterministic split in get_prepared_data ensures these
        # rows match the preprocessed splits main.py passes to train/predict.
        self._raw = get_prepared_data(dataset_name, return_raw=True)
        preprocessor = _build_preprocessor(self._raw[0])

        self._tune_threshold = False
        self.threshold = tuned_threshold if tuned_threshold is not None else 0.5

        if self.task_type == "classification":
            mlp = MLPClassifier(**params, max_iter=2000, early_stopping=True, random_state=42)
            if dataset_name in self.IMBALANCED_DATASETS:
                # 10:1 undersample then 2:1 SMOTE matches mlp_cv.ipynb's strategy
                # for highly imbalanced classification.
                self.model = IMLPipeline([
                    ('preprocess', preprocessor),
                    ('under', RandomUnderSampler(sampling_strategy=0.1, random_state=42)),
                    ('over', SMOTE(sampling_strategy=0.5, random_state=42)),
                    ('mlp', mlp),
                ])
                self._tune_threshold = True
            else:
                self.model = Pipeline([('preprocess', preprocessor), ('mlp', mlp)])
        else:
            mlp = MLPRegressor(**params, max_iter=2000, early_stopping=True, random_state=42)
            self.model = Pipeline([('preprocess', preprocessor), ('mlp', mlp)])

    def train(self, X_train, y_train, X_val, y_val):
        X_tr_raw, X_va_raw, _, y_tr, y_va, _ = self._raw
        self.model.fit(X_tr_raw, y_tr)

        if self._tune_threshold:
            val_prob = self.model.predict_proba(X_va_raw)[:, 1]
            precision, recall, thresholds = precision_recall_curve(y_va, val_prob)
            f1 = 2 * precision * recall / (precision + recall + 1e-12)
            best_idx = int(np.argmax(f1[:-1]))
            self.threshold = thresholds[best_idx]

    def predict(self, X_test):
        X_te_raw = self._raw[2]
        if self._tune_threshold:
            prob = self.model.predict_proba(X_te_raw)[:, 1]
            return (prob >= self.threshold).astype(int)
        return self.model.predict(X_te_raw)
