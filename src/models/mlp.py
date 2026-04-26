from src.models.base_model import BaseModel
from sklearn.neural_network import MLPClassifier, MLPRegressor
import numpy as np

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

        params = self.hyperparameters.get(dataset_name)
        
        if self.task_type == "classification":
            self.model = MLPClassifier(**params, max_iter=2000, random_state=42)
        else:
            self.model = MLPRegressor(**params, max_iter=2000, random_state=42)

    def train(self, X_train, y_train, X_val, y_val):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)