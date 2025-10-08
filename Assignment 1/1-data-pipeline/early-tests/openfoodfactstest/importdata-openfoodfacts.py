# import_openfoodfacts_hf_parquet_to_csv.py
import os
import pandas as pd

# ---------------- CONFIG ----------------
# Requires: pandas >= 2.0, fsspec, huggingface_hub (install if needed)
DATA_DIR = os.path.join("..", "data", "openfoodfacts")
SPLITS = {"food": "food.parquet", "beauty": "beauty.parquet"}
SPLIT_KEY = "food"  # change to "beauty" if you want that split

# ---------------- MAIN ----------------
if __name__ == "__main__":
    hf_path = "hf://datasets/openfoodfacts/product-database/" + SPLITS[SPLIT_KEY]
    print(f"Loading {hf_path} ...")
    df = pd.read_parquet(hf_path)

    os.makedirs(DATA_DIR, exist_ok=True)
    out_csv = os.path.join(DATA_DIR, f"{SPLIT_KEY}.csv")
    df.to_csv(out_csv, index=False)

    print(f"Saved: {out_csv}")
    print(f"All data stored under: {os.path.abspath(DATA_DIR)}")
