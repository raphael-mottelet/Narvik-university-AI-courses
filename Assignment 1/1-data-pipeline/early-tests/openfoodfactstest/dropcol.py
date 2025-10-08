# dropcol.py
#
# Duplicates an OpenFoodFacts-style CSV and keeps ONLY:
#   1) Product metadata needed for the project:
#        - product name, type, category
#   2) Common nutrient columns (fiber, fat, sugar, energy, proteins, salt, etc.)
#   3) Environmental notation columns (ecoscore, carbon footprint)
#   4) One creation-date column ("created_datetime")
#
# Everything else is dropped.
#
# Defaults:
#   --input  ../data/openfoodfacts/en.openfoodfacts.org.products.csv
#   --output ../data/openfoodfacts/en.openfoodfacts.org.products_dropedcol.csv
#
# No parquet logic is used.

import argparse
import os
import sys
from typing import List
import pandas as pd


# ---- product columns to keep (name, type, category) ----
# "type" is best represented by Open Food Facts food group columns.
PRODUCT_META_COLS: List[str] = [
    "product_name",          # product name
    "generic_name",          # optional alt/common name
    # type:
    "food_groups",
    # category:
    "categories",
]

# ---- nutrient columns to keep ----
COMMON_NUTRIENT_COLS: List[str] = [
    # Energy
    "energy-kj_100g", "energy-kcal_100g", "energy_100g",
    # Fats
    "fat_100g", "saturated-fat_100g","cholesterol_100g",
    # Carbs & sugars
    "carbohydrates_100g", "sugars_100g", "added-sugars_100g",
    "sucrose_100g", "glucose_100g", "fructose_100g", "lactose_100g",
    "maltose_100g", "starch_100g",
    # Fiber & protein
    "fiber_100g", "proteins_100g",
    # Salt & sodium
    "salt_100g", "sodium_100g",
    # Alcohol
    "alcohol_100g",
]

# ---- environmental columns to keep ----
ENVIRONMENT_COLS: List[str] = [
    "ecoscore_score",
    "ecoscore_grade",
]

# Optionally keep NutriScore/NOVA (quality/processing indicators, not strictly environmental)
OPTIONAL_QUALITY_GRADE_COLS: List[str] = [
    "nutriscore_score", "nutriscore_grade", "nova_group"
]

# Creation date handling
CREATED_COLS_PRIORITY: List[str] = [
    "created_datetime",  # preferred
]


def parse_args() -> argparse.Namespace:
    default_in = os.path.join("..", "data", "openfoodfacts", "en.openfoodfacts.org.products.csv")
    default_out = os.path.join("..", "data", "openfoodfacts", "en.openfoodfacts.org.products_dropedcol.csv")

    p = argparse.ArgumentParser(
        description="Keep only product name/type/category, nutrients, environmental, and creation-date columns from an OpenFoodFacts CSV."
    )
    p.add_argument("--input", default=default_in, help=f"Path to input CSV. Default: {default_in}")
    p.add_argument("--output", default=default_out, help=f"Path to output CSV. Default: {default_out}")
    p.add_argument(
        "--keep-quality-grades",
        action="store_true",
        help="Also keep NutriScore/NOVA columns (nutriscore_score, nutriscore_grade, nova_group).",
    )
    return p.parse_args()


def ensure_created_datetime(df: pd.DataFrame) -> pd.DataFrame:
    has_created_dt = "created_datetime" in df.columns
    has_created_t = "created_t" in df.columns

    if has_created_dt:
        df["created_datetime"] = df["created_datetime"].astype(str)
        if has_created_t:
            df = df.drop(columns=["created_t"])
        return df
    
    df["created_datetime"] = ""
    return df


def main():
    args = parse_args()

    in_path = args.input
    out_path = args.output

    if not os.path.isfile(in_path):
        print(f"Input file not found: {in_path}", file=sys.stderr)
        sys.exit(2)

    print(f"Loading {in_path} ...")
    df = pd.read_csv(in_path, low_memory=False)

    # Normalize/create creation date
    df = ensure_created_datetime(df)

    # Build keep list
    keep_cols = ["created_datetime"]

    # Product meta (name, type, category)
    keep_cols += [c for c in PRODUCT_META_COLS if c in df.columns]

    # Nutrients + environmental
    keep_cols += [c for c in COMMON_NUTRIENT_COLS if c in df.columns]
    keep_cols += [c for c in ENVIRONMENT_COLS if c in df.columns]

    if args.keep_quality_grades:
        keep_cols += [c for c in OPTIONAL_QUALITY_GRADE_COLS if c in df.columns]

    # Deduplicate while preserving order
    seen = set()
    keep_cols = [c for c in keep_cols if not (c in seen or seen.add(c))]

    # Subset (ensure at least created_datetime)
    present_keep_cols = [c for c in keep_cols if c in df.columns]
    if "created_datetime" not in present_keep_cols:
        present_keep_cols = ["created_datetime"] + present_keep_cols

    df_out = df[present_keep_cols].copy()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_out.to_csv(out_path, index=False)

    print(f"Saved cleaned dataset: {out_path}")
    print(f"Columns kept ({len(df_out.columns)}): {', '.join(df_out.columns)}")


if __name__ == "__main__":
    main()
