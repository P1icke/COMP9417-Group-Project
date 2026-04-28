Instructions

Setup:
1. Make sure you have uv installed. Ask ChatGPT if you don't have it installed and don't know how
2. Run "uv venv" to start the virtual environment
3. Run "uv sync" to update dependencies

Kaggle:
1. Needs an API token. Login to www.kaggle.com, go to settings and create an api token.
2. create a .env file in the root directory. add and fill in:
   KAGGLE_USERNAME=xxxxxx
   KAGGLE_KEY=xxxxxx
4. Run "uv run python download_data.py"
5. Run "uv run python src/data_processor.py"

Testing:
1. Run the entire program with "uv run python main.py" (don't recommend this yet because one of the models is slow)
2. Run on a single model with flag --algo, e.g "uv run python main.py --algo mlp"
3. Run on a single dataset with flag --dataset, e.g "uv run python main.py --dataset regression_mixed"

Experiments:
Tuners, trainers and analysis scripts live under experiments/ and are invoked as modules so imports from src/ resolve correctly.

1. Tune xRFM: "uv run python -m experiments.tune_xrfm"
   Writes winners to tuned_params/xrfm/<dataset>.json.
2. Tune XGBoost: "uv run python -m experiments.tune_xgboost"
   Writes winners to tuned_params/xgboost/<dataset>.json.
3. Run XGBoost across all datasets: "uv run python -m experiments.train_xgboost"
   Writes per-dataset metrics to results/xgboost_output.txt.
4. Extract xRFM AGOPs for interpretability: "uv run python -m experiments.analyse_agops"
   Writes labelled AGOPs to results/agops/<dataset>.pt.
5. View saved AGOPs (text summary + plots): "uv run python -m experiments.view_agops"
   Writes per-dataset bar charts and (where available) heatmaps to results/agops/plots/.
   Flags: --dataset <name> for one dataset, --top N for top-N features, --no-plots for text only.

Model wrappers auto-load tuned_params/<model>/<dataset>.json when present and fall back to hardcoded defaults otherwise, so run the tuners before the trainers to get the intended results.

AGOP findings (top 3 features per dataset, normalised diagonal importance from results/agops/<dataset>.pt):

| Dataset | Task | Top 3 features (importance) | AGOP storage |
|---|---|---|---|
| Regression_Mixed (Insurance) | regression | smoker_yes (1.00), smoker_no (1.00), bmi (0.42) | full (11, 11) |
| Regression_d_gt_50 (Ames) | regression | TotalSF (1.00), Overall Qual (0.10), Gr Liv Area (0.08) | diag-only (68,) |
| Regression_n_gt_10k (CASP) | regression | F5 (1.00), F1 (0.57), F2 (0.23) | full (9, 9) |
| Classification_Mixed (Heart Failure) | binary | ST_Slope_Up (1.00), ST_Slope_Flat (0.73), ChestPainType_ASY (0.55) | diag-only (20,) |
| Classification_d_gt_50 (Bankruptcy) | binary | Persistent EPS (1.00), Borrowing dependency (0.96), ROA(A) (0.68) | diag-only (95,) |
| Classification_n_gt_10k (Online Shoppers) | binary | PageValues (1.00), ExitRates (0.004), BounceRates (0.003) | full (28, 28) |

Datasets whose tuned params chose `diag=True` store only the AGOP diagonal as a 1-D vector; datasets with `diag=False` store the full (d, d) matrix and so support off-diagonal interaction analysis. The viewer handles both shapes automatically.


