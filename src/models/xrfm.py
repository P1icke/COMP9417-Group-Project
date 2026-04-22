from src.models.base_model import BaseModel
from xrfm import xRFM


class xRFMAlgorithm(BaseModel):
    def __init__(self, dataset_name, task_type):
        super().__init__(dataset_name, task_type)

        tuning_metric = "mse" if task_type == "regression" else "accuracy"

        self.model = xRFM(
            tuning_metric=tuning_metric,
            random_state=42,
        )

    def train(self, X_train, y_train, X_val, y_val):
        self.model.fit(X_train, y_train, X_val, y_val)

    def predict(self, X_test):
        return self.model.predict(X_test)
