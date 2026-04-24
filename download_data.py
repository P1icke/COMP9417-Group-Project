import os
import zipfile
import subprocess
import glob
from dotenv import load_dotenv

load_dotenv()

os.environ['KAGGLE_USERNAME'] = os.getenv('KAGGLE_USERNAME')
os.environ['KAGGLE_KEY'] = os.getenv('KAGGLE_KEY')

DATA_DIR = "./data"

datasets_map = {
    "Regression_n_gt_10k": {
        "slug": "samanemami/properties-of-protein-tertiary-structure",
    },
    "Regression_d_gt_50": {
        "slug": "fatemeengashte/ameshousing-engineered",
    },
    "Regression_Mixed": {
        "slug": "mirichoi0218/insurance",
    },
    "Classification_n_gt_10k": {
        "slug": "henrysue/online-shoppers-intention",
    },
    "Classification_d_gt_50": {
        "slug": "fedesoriano/company-bankruptcy-prediction",
    },
    "Classification_Mixed": {
        "slug": "fedesoriano/heart-failure-prediction",
    }
}

def download_and_extract(name, info):
    slug = info['slug']
    extract_path = os.path.join(DATA_DIR, name)
    os.makedirs(extract_path, exist_ok=True)
    
    print(f"Downloading {name} ({slug})...")
    command = f"kaggle datasets download -d {slug} -p {extract_path}"
    
    try:
        subprocess.run(command.split(), check=True, capture_output=True)

        for file in os.listdir(extract_path):
            if file.endswith(".zip"):
                zip_path = os.path.join(extract_path, file)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
                os.remove(zip_path)
        
        csv_files = glob.glob(os.path.join(extract_path, "*.csv"))
        
        if not csv_files:
            print(f" No CSV found for {name}.")
            return
            
        if info.get("has_labels"):
            for csv in csv_files:
                if "label" in csv.lower():
                    os.rename(csv, os.path.join(extract_path, f"{name}_labels.csv"))
                elif "data" in csv.lower():
                    os.rename(csv, os.path.join(extract_path, f"{name}.csv"))
            print(f" Successfully processed {name} (Features + Labels)")
            return

        main_csv = max(csv_files, key=os.path.getsize)
        new_path = os.path.join(extract_path, f"{name}.csv")
        
        if main_csv != new_path:
            if os.path.exists(new_path):
                os.remove(new_path)
            os.rename(main_csv, new_path)
            
        print(f" Successfully saved as: {new_path}")

    except Exception as e:
        print(f" Failed to process {name}: {e}")

if __name__ == "__main__":
    for name, info in datasets_map.items():
        download_and_extract(name, info)