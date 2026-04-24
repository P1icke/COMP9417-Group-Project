#Tune for optimal tree values for each dataset.

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

with open('results/tuning_xgboost.txt', 'a') as f:
    f.write(f'\n===XGboost model output - {time.ctime()}===\n')

    data = {("Classification_d_gt_50", 'classification'),
            ("Classification_Mixed", 'classification'),
            ("Classification_n_gt_10k", 'classification'),
            ("Regression_d_gt_50", 'regression'),
            ("Regression_Mixed", 'regression'),
            ("Regression_n_gt_10k", 'regression')}


    for name, type in data:
        f.write(f'\nDATASET:{name}\n')
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
        
        #calculate inference time (avg time per sample)
        output['Inference time per sample (s)'] = round((output['Training Time (s)']/len(X_train)), 5)
        
        #Measure AUC for all data splits, to see if any is particularly overfit
        if "Classification" in name:

            train_probs = model.model.predict_proba(X_train)[:, 1]
            test_probs = model.model.predict_proba(X_test)[:, 1]
            val_probs = model.model.predict_proba(X_val)[:, 1]

            train_auc = round(roc_auc_score(y_train, train_probs), 4)
            test_auc = round(roc_auc_score(y_test, test_probs), 4)
            val_auc = round(roc_auc_score(y_val, val_probs), 4)
            f.write(f'train_auc: {train_auc}\n')
            f.write(f'val_auc: {val_auc}\n')
            f.write(f'test_auc: {test_auc}\n')
           

        else:
            y_mean = y_train.mean()
            f.write(f'mean of y_train: {y_mean}\n')
            
        f.write(f'General output\n{output}\n')
    


#Notes

#CLass. n > 10k
#Train AUC > test AUC -> potential overfitting
#   - could reduce tree complexity
#   - 

#Class. d > 50
#High overfitting!
#   - adjust complexity: pruning? depth?

#Class. mixed
#Train AUC > test AUC  -> potential overfitting


#Regression models all helpful - RMSE significantly lower than simple mean
#Could be helpful to display metrics for train_test_val like for AUC.
#   - means to compare against RMSE.


#Reg. n > 10k
#Should train over a range of tree-depth {4-10} to see where RMSE performs best

#Reg. d > 50
#See above, {4-11} (potential complex interactions)

#Reg. mixed


