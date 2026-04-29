#!/bin/bash
# Re-run all four models across all six datasets, capturing each model's
# output to its own txt file. xRFM is run last because CASP dominates the
# total wall-clock time (~11 min); the other three models complete in
# seconds, so MLP / RandomForest / XGBoost numbers are visible early.

set -u

LOG_DIR="results/run_logs"
mkdir -p "$LOG_DIR"

# Start with a clean CSV so the run produces an internally consistent
# benchmark_log.csv from a single sweep.
rm -f results/benchmark_log.csv

echo "=== Full benchmark re-run started at $(date) ==="
echo "Logs will land in $LOG_DIR/"
echo

echo "--- MLP ---"
caffeinate -i uv run python -u main.py --algo mlp 2>&1 | tee "$LOG_DIR/run_mlp.txt"
echo

echo "--- RandomForest ---"
caffeinate -i uv run python -u main.py --algo randomforest 2>&1 | tee "$LOG_DIR/run_randomforest.txt"
echo

echo "--- XGBoost ---"
caffeinate -i uv run python -u main.py --algo xgboost 2>&1 | tee "$LOG_DIR/run_xgboost.txt"
echo

echo "--- xRFM (longest; CASP alone is ~11 min) ---"
caffeinate -i uv run python -u main.py --algo xrfm 2>&1 | tee "$LOG_DIR/run_xrfm.txt"
echo

echo "=== Full benchmark re-run finished at $(date) ==="
echo "Run summaries in $LOG_DIR/run_<model>.txt"
echo "Combined CSV: results/benchmark_log.csv"
