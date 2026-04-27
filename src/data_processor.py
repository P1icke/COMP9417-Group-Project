import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

DATA_DIR = "./data"

DATASET_CONFIG = {
    "Regression_n_gt_10k": {
        "folder": "Regression_n_gt_10k",
        "file": "Regression_n_gt_10k.csv",
        "target": "RMSD", 
        "drop_cols": [],
        "task": "regression"
    },
    "Regression_d_gt_50": {
        "folder": "Regression_d_gt_50",
        "file": "Regression_d_gt_50.csv",
        "target": "SalePrice", 
        "drop_cols": [], 
        "task": "regression"
    },
    "Regression_Mixed": {
        "folder": "Regression_Mixed",
        "file": "Regression_Mixed.csv",
        "target": "charges", 
        "drop_cols": [],
        "task": "regression"
    },
    "Classification_n_gt_10k": {
        "folder": "Classification_n_gt_10k",
        "file": "Classification_n_gt_10k.csv",
        "target": "Class",
        "drop_cols": [],
        "task": "classification"
    },
    "Classification_d_gt_50": {
        "folder": "Classification_d_gt_50",
        "file": "Classification_d_gt_50.csv",
        "target": "Bankrupt?", 
        "drop_cols": [], 
        "task": "classification"
    },
    "Classification_Mixed": {
        "folder": "Classification_Mixed",
        "file": "Classification_Mixed.csv",
        "target": "HeartDisease", 
        "drop_cols": [],
        "task": "classification"
    }
}

def _build_preprocessor(X):
    """Dynamically builds a scikit-learn ColumnTransformer based on column datatypes."""
    numeric_features = X.select_dtypes(include=['int64', 'float64', 'bool']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category', 'str', 'string']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def get_prepared_data(dataset_name, val_size=0.15, test_size=0.15, random_state=42):
    """
    Loads, cleans, splits, and preprocesses a specified dataset into Train, Val, and Test sets.
    """
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Dataset '{dataset_name}' not found in configuration.")
        
    config = DATASET_CONFIG[dataset_name]
    file_path = os.path.join(DATA_DIR, config["folder"], config["file"])
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file missing: {file_path}. Please run the download script first.")
        
    df = pd.read_csv(file_path)
        
    cols_to_drop = [col for col in config["drop_cols"] if col in df.columns]
    df = df.drop(columns=cols_to_drop)
    
    X = df.drop(columns=[config["target"]])
    y = df[config["target"]]
    
    temp_size = val_size + test_size
    stratify_col = y if config["task"] == "classification" else None
    
    X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
        X, y, test_size=temp_size, random_state=random_state, stratify=stratify_col
    )
    
    test_ratio = test_size / temp_size
    stratify_col_temp = y_temp if config["task"] == "classification" else None
    
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp_raw, y_temp, test_size=test_ratio, random_state=random_state, stratify=stratify_col_temp
    )
    
    preprocessor = _build_preprocessor(X_train_raw)
    
    X_train = preprocessor.fit_transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    X_test = preprocessor.transform(X_test_raw)
    
    if y_train.dtype == 'object' or y_train.dtype.name == 'category':
        y_train = y_train.astype('category').cat.codes
        y_val = y_val.astype('category').cat.codes
        y_test = y_test.astype('category').cat.codes
    elif y_train.dtype == 'bool':
        y_train = y_train.astype(int)
        y_val = y_val.astype(int)
        y_test = y_test.astype(int)

    return X_train, X_val, X_test, y_train.values, y_val.values, y_test.values


def get_feature_names(dataset_name, val_size=0.15, test_size=0.15, random_state=42):
    """Return the column names produced by get_prepared_data's preprocessor.

    Replays the same split + preprocessor fit so one-hot category expansions
    match what the models were trained on.
    """
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Dataset '{dataset_name}' not found in configuration.")

    config = DATASET_CONFIG[dataset_name]
    file_path = os.path.join(DATA_DIR, config["folder"], config["file"])
    df = pd.read_csv(file_path)
    df = df.drop(columns=[c for c in config["drop_cols"] if c in df.columns])

    X = df.drop(columns=[config["target"]])
    y = df[config["target"]]

    stratify = y if config["task"] == "classification" else None
    X_train_raw, _, _, _ = train_test_split(
        X, y, test_size=val_size + test_size, random_state=random_state, stratify=stratify
    )

    preprocessor = _build_preprocessor(X_train_raw)
    preprocessor.fit(X_train_raw)
    return preprocessor.get_feature_names_out().tolist()


if __name__ == "__main__":
    print("Running 3-way split processor test suite...\n")
    
    for test_dataset in DATASET_CONFIG.keys():
        try:
            X_tr, X_va, X_te, y_tr, y_va, y_te = get_prepared_data(test_dataset)
            print(f"Successfully processed {test_dataset}")
            print(f"   Train: {X_tr.shape[0]} rows | Val: {X_va.shape[0]} rows | Test: {X_te.shape[0]} rows")
        except Exception as e:
            print(f"Test failed for {test_dataset}: {e}")