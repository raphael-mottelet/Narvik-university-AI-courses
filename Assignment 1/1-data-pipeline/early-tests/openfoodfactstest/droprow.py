# droprow.py
#
# Drops rows that are missing ANY of the required columns (after normalizing empties to NaN).
# Defaults:
#   --input  ../data/openfoodfacts/food_fr_clean_col.csv
#   --output ../data/openfoodfacts/food_fr_required.csv
#
# Required columns (can be overridden with --required):
#   ecoscore_score, ecoscore_grade, energy-kj_100g, cholesterol_100g, fiber_100g,
#   food_groups, food_groups_tags, categories, categories_tags, salt_100g,
#   sodium_100g, saturated-fat_100g
#
# Usage (PowerShell):
#   py.exe .\droprow.py --input ..\data\openfoodfacts\food_fr_clean_col.csv --output ..\data\openfoodfacts\food_fr_required.csv
#
# No parquet logic. Pure CSV + pandas.

import argparse
import os
import sys
import math
from typing import List, Any

import numpy as np
import pandas as pd

DEFAULT_INPUT = os.path.join("..", "data", "openfoodfacts", "food_fr_clean_col.csv")
DEFAULT_OUTPUT = os.path.join("..", "data", "openfoodfacts", "food_fr_required.csv")

DEFAULT_REQUIRED: List[str] = [
    "ecoscore_score",
    "ecoscore_grade",
    "energy-kj_100g",
    "cholesterol_100g",
    "fiber_100g",
    "food_groups",
    "food_groups_tags",
    "categories",
    "categories_tags",
    "salt_100g",
    "sodium_100g",
    "saturated-fat_100g",
    "food_groups_tags",
    "categories_tags",
]

def normalize_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize common 'empty' values to NaN:
      - None, NaN
      - "" / whitespace-only
      - "nan" (case-insensitive)
      - "[]" / "{}"
    """
    def _norm(x: Any):
        if x is None:
            return np.nan
        if isinstance(x, float) and math.isnan(x):
            return np.nan
        if isinstance(x, str):
            s = x.strip()
            if s == "" or s.lower() == "nan" or s in ("[]", "{}"):
                return np.nan
            return x
        return x
    return df.applymap(_norm)

def droprow(df: pd.DataFrame, required_cols: List[str]) -> pd.DataFrame:
    """
    Drop rows if ANY of required_cols is missing (NaN after normalization).
    Keeps only rows where all required columns are non-missing.
    """
    present = [c for c in required_cols if c in df.columns]
    if not present:
        return df.copy()

    mask_all_present = df[present].notna().all(axis=1)
    return df.loc[mask_all_present].copy()

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Drop rows missing any of the required columns.")
    p.add_argument("--input", default=DEFAULT_INPUT, help=f"Path to input CSV. Default: {DEFAULT_INPUT}")
    p.add_argument("--output", default=DEFAULT_OUTPUT, help=f"Path to output CSV (filtered). Default: {DEFAULT_OUTPUT}")
    p.add_argument(
        "--required",
        default=",".join(DEFAULT_REQUIRED),
        help="Comma-separated list of required columns. Defaults to built-in list.",
    )
    return p.parse_args()

def main():
    args = parse_args()
    in_path = args.input
    out_path = args.output
    required_cols = [c.strip() for c in args.required.split(",") if c.strip()]

    if not os.path.isfile(in_path):
        print(f"Input file not found: {in_path}", file=sys.stderr)
        sys.exit(2)

    print(f"Loading {in_path} ...")
    df = pd.read_csv(in_path, low_memory=False)
    before_rows = len(df)

    # Normalize empties -> NaN, then drop rows missing any required values
    df = normalize_missing(df)

    # Warn if some required columns are absent
    missing_required_cols = [c for c in required_cols if c not in df.columns]
    if missing_required_cols:
        print("WARNING: These required columns are not in the file and will be ignored:")
        for c in missing_required_cols:
            print(f"  - {c}")

    df_filtered = droprow(df, required_cols)

    dropped = before_rows - len(df_filtered)
    pct = (dropped / before_rows * 100) if before_rows else 0.0

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df_filtered.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(f"Rows before: {before_rows:,}")
    print(f"Rows after : {len(df_filtered):,}")
    print(f"Dropped    : {dropped:,} ({pct:.1f}%)")
    print("Done.")

if __name__ == "__main__":
    main()
