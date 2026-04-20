from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, dataset_name, task_type, **kwargs):
        self.model = None
        self.dataset_name = dataset_name
        self.task_type = task_type
        
    @abstractmethod
    def train(self, X_train, y_train, X_val, y_val):
        """
        Fit the machine learning model here.
        Use X_val and y_val for early stopping or manual validation checks.
        """
        pass
        
    @abstractmethod
    def predict(self, X_test):
        """Return the array of predictions here."""
        pass