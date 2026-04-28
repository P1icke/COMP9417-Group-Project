import argparse
import os
    
import pandas as pd
import datetime as dt

from src.data_processor import get_prepared_data, DATASET_CONFIG
from src.evaluator import evaluate_model
from src.models.mlp import MLPAlgorithm
from src.models.xgboost import XGBoostAlgorithm
from src.models.xrfm import xRFMAlgorithm
from src.models.random_forest import RandomForestAlgorithm

def main():
    parser = argparse.ArgumentParser(description="Run ML Benchmark Pipeline")
    parser.add_argument("--algo", type=str, help="Run a specific algorithm (e.g., 'MLP')")
    parser.add_argument("--dataset", type=str, help="Run a specific dataset (e.g., 'Regression_Mixed')")
    args = parser.parse_args()
    log_columns_names = [
        'Dataset',
        'Algorithm',
        'Metric Type',
        'Test Score',
        'Training Time (s)',
        'Inference Time per sample (s)',
    ]

    os.makedirs("results", exist_ok=True)
    all_results = []

    print("Starting Benchmark Pipeline...\n")

    datasets_to_run = DATASET_CONFIG.items()
    if args.dataset:
        datasets_to_run = {k: v for k, v in DATASET_CONFIG.items() if args.dataset.lower() in k.lower()}.items()
        
        if not datasets_to_run:
            print(f" Error: Could not find any dataset matching '{args.dataset}'")
            return

    for dataset_name, config in datasets_to_run:
        print(f"Loading and processing: {dataset_name}...")
        try:
            X_tr, X_va, X_te, y_tr, y_va, y_te = get_prepared_data(dataset_name)
        except Exception as e:
            print(f"Skipping {dataset_name} due to error: {e}")
            continue

        task = config["task"]

        models_to_run = {
            "MLP": MLPAlgorithm(dataset_name=dataset_name, task_type=task),
            "XGBoost": XGBoostAlgorithm(dataset_name=dataset_name, task_type=task),
            "xRFM": xRFMAlgorithm(dataset_name=dataset_name, task_type=task),
            "RandomForest": RandomForestAlgorithm(dataset_name=dataset_name, task_type=task),
        }

        if args.algo:
            models_to_run = {k: v for k, v in models_to_run.items() if args.algo.lower() in k.lower()}
            
            if not models_to_run:
                print(f"Error: Could not find any algorithm matching '{args.algo}'")
                continue

        for algo_name, model_instance in models_to_run.items():
            if hasattr(model_instance, 'IS_PLACEHOLDER') and model_instance.IS_PLACEHOLDER:
                print(f"  -> {algo_name} (PLACEHOLDER - not implemented)")
            else:
                print(f"  -> Training {algo_name}...")
            
            result = evaluate_model(
                model_instance=model_instance,
                X_train=X_tr, y_train=y_tr,
                X_val=X_va, y_val=y_va,
                X_test=X_te, y_test=y_te,
                dataset_name=dataset_name,
                algorithm_name=algo_name
            )
            
            if result:
                all_results.append(result)
                # For each new test, create a new benchmark log
                # In case of a crash, append the CSV so we still get partial results
                csv_path = f'results/benchmark_log_{dt.datetime.now().isoformat()}.csv'
                pd.DataFrame([result]).to_csv(
                    csv_path,
                    mode="w",
                    header=log_columns_names,
                    index=False,
                )


    if all_results:
        results_df = pd.DataFrame(all_results)
        print("\nRun Summary:\n")
        print(results_df.to_string(index=False))

        placeholder_count = (results_df["Test Score"] == "[PLACEHOLDER]").sum()
        if placeholder_count > 0:
            print(f"\nNote: {placeholder_count} model(s) are placeholders and have not been implemented yet.")

if __name__ == "__main__":
    main()