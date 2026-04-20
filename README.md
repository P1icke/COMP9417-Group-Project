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


