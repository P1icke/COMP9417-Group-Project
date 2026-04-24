from src.models.xgboost import XGBoostAlgorithm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import time
import numpy as np
from src.evaluator import evaluate_model
from src.data_processor import get_prepared_data
import sys


np.random.seed(42)

with open('results/xgboost_output.txt', 'w') as f:
    f.write(f'XGboost model output - {time.ctime()}\n\n')

data = [("Classification_d_gt_50", 'classification'),
        ("Classification_Mixed", 'classification'),
        ("Classification_n_gt_10k", 'classification'),
        ("Regression_d_gt_50", 'regression'),
        ("Regression_Mixed", 'regression'),
        ("Regression_n_gt_10k", 'regression')]


for name, type in data:
    model = XGBoostAlgorithm(name, type)
    X_train, X_val, X_test, y_train, y_val, y_test = get_prepared_data(name)

    output = evaluate_model(model,
                            X_train,
                            y_train,
                            X_val,
                            y_val,
                            X_test,
                            y_test,
                            name,
                            "XGBoost")
    
    #calculate inference time (avg time per sample) PLACEHOLDER because 
    #should be done in evaluator.py
    output['PLACEHOLDER Inference time/sample (s)'] = round((output['Training Time (s)']/len(X_train)), 5)
    
    if "Classification" in name:

        probs = model.model.predict_proba(X_test)[:, 1]
        output["AUC-ROC"] = round(roc_auc_score(y_test, probs), 4)
    

    with open('results/xgboost_output.txt', 'a') as f:
        f.write(f'{output}\n')
    
#need to store output in a file.
