from src.models.base_model import BaseModel
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np

class RandomForestAlgorithm(BaseModel):
    def __init__(self, dataset_name, task_type):
        super().__init__(dataset_name, task_type)      
        self.hyperparameters = {
            "Regression_n_gt_10k":      {"n_estimators": 100, "max_depth": 20, "min_samples_split": 5, "min_samples_leaf": 2},
            "Regression_d_gt_50":       {"n_estimators": 100, "max_depth": 20, "min_samples_split": 5, "min_samples_leaf": 2},
            "Regression_Mixed":         {"n_estimators": 100, "max_depth": 20, "min_samples_split": 5, "min_samples_leaf": 2},
            "Classification_n_gt_10k":  {"n_estimators": 100, "max_depth": 20, "min_samples_split": 5, "min_samples_leaf": 2},
            "Classification_d_gt_50":   {"n_estimators": 100, "max_depth": 20, "min_samples_split": 5, "min_samples_leaf": 2},
            "Classification_Mixed":     {"n_estimators": 100, "max_depth": 20, "min_samples_split": 5, "min_samples_leaf": 2},
        }
        
        params = self.hyperparameters.get(dataset_name)
        
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