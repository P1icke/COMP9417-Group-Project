from src.models.base_model import BaseModel
from xgboost import XGBClassifier, XGBRegressor

class XGBoostAlgorithm(BaseModel):
    def __init__(self, dataset_name, task_type):
        super().__init__(dataset_name, task_type)
        self.hyperparameters = {
            "Regression_n_gt_10k": {
                "n_estimators": 1000, 
                "learning_rate": 0.05, 
                "max_depth": 6, 
                "subsample": 0.8
            },
            "Regression_d_gt_50": {
                "n_estimators": 500, 
                "learning_rate": 0.1, 
                "max_depth": 8, 
                "colsample_bytree": 0.7
            },
            "Regression_Mixed": {
                "n_estimators": 300, 
                "learning_rate": 0.01, 
                "max_depth": 4
            },
            "Classification_n_gt_10k": {
                "n_estimators": 1000,
                "learning_rate": 0.1,
                "max_depth": 5,
            },
            "Classification_d_gt_50": {
                "n_estimators": 400, 
                "learning_rate": 0.05, 
                "max_depth": 10, 
                "colsample_bylevel": 0.6
            },
            "Classification_Mixed": {
                "n_estimators": 200, 
                "learning_rate": 0.1, 
                "max_depth": 3
            }
        }
        
        params = self.hyperparameters.get(dataset_name)
        
        if self.task_type == "classification":
            self.model = XGBClassifier(
                **params, 
                early_stopping_rounds=10, 
                random_state=42, 
                eval_metric='logloss'
            )
        else:
            self.model = XGBRegressor(
                **params, 
                early_stopping_rounds=10, 
                random_state=42
            )

    def train(self, X_train, y_train, X_val, y_val):
        if self.task_type == "classification":
            pos = int((y_train == 1).sum())
            neg = int((y_train == 0).sum())
            if pos > 0:
                self.model.set_params(scale_pos_weight=neg / pos)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

    def predict(self, X_test):
        return self.model.predict(X_test)