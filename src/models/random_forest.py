import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.models.base_model import BaseModel


class RandomForestAlgorithm(BaseModel):
    DEFAULT_PARAMS = {
        "n_estimators": 100,
        "max_depth": 20,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
    }

    def __init__(self, dataset_name, task_type):
        super().__init__(dataset_name, task_type)

        tuned_path = Path(f"tuned_params/random_forest/{dataset_name}.json")
        if tuned_path.exists():
            with open(tuned_path) as f:
                params = json.load(f)["params"]
        else:
            params = dict(self.DEFAULT_PARAMS)
        
        if self.task_type == "classification":
            self.model = RandomForestClassifier(
                **params,
                random_state=42,
                n_jobs=-1
            )
        else:
            self.model = RandomForestRegressor(
                **params,
                random_state=42,
                n_jobs=-1
            )

    def train(self, X_train, y_train, X_val, y_val):
        y_train_array = np.array(y_train).flatten()
        self.model.fit(X_train, y_train_array)

    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        if self.task_type == "classification":
            return self.model.predict_proba(X_test)
        raise NotImplementedError("predict_proba is only available for classification tasks")