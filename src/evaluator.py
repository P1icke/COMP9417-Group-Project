# src/evaluator.py
import time
from sklearn.metrics import accuracy_score, root_mean_squared_error

def evaluate_model(model_instance, X_train, y_train, X_val, y_val, X_test, y_test, dataset_name, algorithm_name):
    """Standardized evaluation wrapper ensuring no data leakage."""
    
    if hasattr(model_instance, 'IS_PLACEHOLDER') and model_instance.IS_PLACEHOLDER:
        log_entry = {
            "Dataset": dataset_name,
            "Algorithm": algorithm_name,
            "Metric Type": "N/A",
            "Test Score": "[PLACEHOLDER]",
            "Training Time (s)": 0.0
        }
        return log_entry
    
    start_time = time.time()
    
    try:
        model_instance.train(X_train, y_train, X_val, y_val)
    except Exception as e:
        print(f"Error training {algorithm_name} on {dataset_name}: {e}")
        return None
        
    train_time = time.time() - start_time
    
    try:
        predictions = model_instance.predict(X_test)
    except Exception as e:
        print(f"Error predicting with {algorithm_name} on {dataset_name}: {e}")
        return None
    
    if "Classification" in dataset_name:
        score = accuracy_score(y_test, predictions)
        metric_name = "Accuracy"
    else:
        score = root_mean_squared_error(y_test, predictions)
        metric_name = "RMSE"
        
    log_entry = {
        "Dataset": dataset_name,
        "Algorithm": algorithm_name,
        "Metric Type": metric_name,
        "Test Score": round(score, 4),
        "Training Time (s)": round(train_time, 4)
    }
    
    return log_entry