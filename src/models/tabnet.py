from src.models.base_model import BaseModel
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
import numpy as np

class TabNetAlgorithm(BaseModel):
    def __init__(self, dataset_name, task_type):
        super().__init__(dataset_name, task_type)      
        self.hyperparameters = {
            "Regression_n_gt_10k":      {"n_steps": 2, "n_independent": 2, "n_shared": 1},
            "Regression_d_gt_50":       {"n_steps": 2, "n_independent": 2, "n_shared": 1},
            "Regression_Mixed":         {"n_steps": 2, "n_independent": 2, "n_shared": 1},
            "Classification_n_gt_10k":  {"n_steps": 3, "n_independent": 2, "n_shared": 2},
            "Classification_d_gt_50":   {"n_steps": 3, "n_independent": 2, "n_shared": 2},
            "Classification_Mixed":     {"n_steps": 3, "n_independent": 2, "n_shared": 2},
        }
        
        params = self.hyperparameters.get(dataset_name)
        
        if self.task_type == "classification":
            self.model = TabNetClassifier(
                **params,
                verbose=0,
                seed=42
            )
        else:
            self.model = TabNetRegressor(
                **params,
                verbose=0,
                seed=42
            )

    def train(self, X_train, y_train, X_val, y_val):
        if self.task_type == "regression":
            y_train_reshaped = np.array(y_train).reshape(-1, 1) if len(np.array(y_train).shape) == 1 else y_train
            y_val_reshaped = np.array(y_val).reshape(-1, 1) if len(np.array(y_val).shape) == 1 else y_val
        else:
            y_train_reshaped = np.array(y_train).flatten()
            y_val_reshaped = np.array(y_val).flatten()
        
        self.model.fit(
            X_train, y_train_reshaped,
            eval_set=[(X_val, y_val_reshaped)],
            patience=20,
            max_epochs=150
        )

    def predict(self, X_test):
        return self.model.predict(X_test)