# import-data.py
import json, pathlib, requests, shutil, sys

ROOT = pathlib.Path("../../data")
MATV = ROOT / "matvaretabellen"
KAGG = ROOT / "kaggle" / "foodcom"
MATV.mkdir(parents=True, exist_ok=True)
KAGG.mkdir(parents=True, exist_ok=True)

BASE = "https://www.matvaretabellen.no"
ENDPOINTS = {
    "foods_en":       "/api/en/foods.json",
    "food_groups_en": "/api/en/food-groups.json",
    "nutrients_en":   "/api/en/nutrients.json",
    "sources_en":     "/api/en/sources.json",
}

def import_matvaretabellen():
    for name, path in ENDPOINTS.items():
        r = requests.get(BASE + path, timeout=60)
        r.raise_for_status()
        outp = MATV / f"{name}.json"
        with open(outp, "w", encoding="utf-8") as f:
            json.dump(r.json(), f, ensure_ascii=False, indent=2)
        print(f"saved: {outp.resolve()}")

def import_kaggle_foodcom_csv_only():
    try:
        import kagglehub  # pip install kagglehub
    except Exception:
        print("Missing dependency. Install with: pip install kagglehub")
        sys.exit(1)

    src_root = pathlib.Path(kagglehub.dataset_download("irkaal/foodcom-recipes-and-reviews"))

    copied = []
    for p in src_root.rglob("*"):
        if p.is_file() and p.suffix.lower() == ".csv":
            dest = KAGG / p.name  # flatten
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)  # overwrite without deleting dir
            copied.append(dest.resolve())

    if not copied:
        print("WARNING: no CSV files found in the Kaggle dataset.")
    else:
        print("CSV files copied to:", KAGG.resolve())
        for c in copied:
            size = c.stat().st_size
            print(f" - {c}  ({size:,} bytes)")

def main():
    import_matvaretabellen()
    import_kaggle_foodcom_csv_only()
    print("All data under:", ROOT.resolve())

if __name__ == "__main__":
    main()
