import torch
from pathlib import Path

from src.data_processor import get_prepared_data, get_feature_names, DATASET_CONFIG
from src.models.xrfm import xRFMAlgorithm


AGOPS_DIR = Path("results/agops")
AGOPS_DIR.mkdir(parents=True, exist_ok=True)


for dataset_name, config in DATASET_CONFIG.items():
    out_path = AGOPS_DIR / f"{dataset_name}.pt"
    if out_path.exists():
        print(f"Skipping {dataset_name} (already analyzed)")
        continue

    print(f"\n=== {dataset_name} ===")
    X_tr, X_va, X_te, y_tr, y_va, y_te = get_prepared_data(dataset_name)

    # xRFMAlgorithm picks up tuned_params/xrfm/{name}.json automatically if present.
    model = xRFMAlgorithm(dataset_name=dataset_name, task_type=config["task"])
    model.train(X_tr, y_tr, X_va, y_va)

    agops = model.get_leaf_agops()
    feature_names = get_feature_names(dataset_name)
    torch.save({"agops": agops, "feature_names": feature_names}, out_path)
    print(f"  saved {len(agops)} AGOP matrices ({len(feature_names)} features) to {out_path}")
