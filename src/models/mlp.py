from src.models.base_model import BaseModel
from sklearn.neural_network import MLPClassifier, MLPRegressor

class MLPAlgorithm(BaseModel):
    def __init__(self, dataset_name, task_type):
        super().__init__(dataset_name, task_type)
        self.hyperparameters = {
            "Regression_n_gt_10k":      {"hidden_layer_sizes": (100, 50), "alpha": 0.01},
            "Regression_d_gt_50":       {"hidden_layer_sizes": (200,), "alpha": 0.001},
            "Regression_Mixed":         {"hidden_layer_sizes": (50,), "alpha": 0.0001},
            "Classification_n_gt_10k":  {"hidden_layer_sizes": (100, 50, 25), "alpha": 0.01},
            "Classification_d_gt_50":   {"hidden_layer_sizes": (50, 25), "alpha": 0.05},
            "Classification_Mixed":     {"hidden_layer_sizes": (10,), "alpha": 0.1},
        }
        
        params = self.hyperparameters.get(dataset_name)
        
        if self.task_type == "classification":
            self.model = MLPClassifier(**params, max_iter=2000, random_state=42)
        else:
            self.model = MLPRegressor(**params, max_iter=2000, random_state=42)

    def train(self, X_train, y_train, X_val, y_val):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        if self.task_type == "classification":
            return self.model.predict_proba(X_test)
        raise NotImplementedError("predict_proba is only available for classification tasks")