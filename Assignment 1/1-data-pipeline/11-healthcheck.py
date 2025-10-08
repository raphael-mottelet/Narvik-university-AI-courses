#!/usr/bin/env python3
# health_check_step10.py
# Checks nulls in price_nok_total & carbon_kg_total and reports the most
# expensive items by product_name in NOK and CO2 (kg).
#
# Usage:
#   python health_check_step10.py --input ../data/openfoodfacts/step10.csv

import os, sys, argparse
import pandas as pd

DEF_IN = os.path.join("..", "data", "openfoodfacts", "step10.csv")

def main():
    ap = argparse.ArgumentParser(description="Health check for price/carbon totals in step10 dataset.")
    ap.add_argument("--input", default=DEF_IN, help=f"Input CSV (default: {DEF_IN})")
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(2)

    df = pd.read_csv(args.input, low_memory=False)

    # Ensure required columns
    req = ["product_name", "price_nok_total", "carbon_kg_total"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        print(f"Missing required columns: {missing}", file=sys.stderr)
        sys.exit(2)

    # Coerce numeric for diagnostics
    df["price_nok_total"] = pd.to_numeric(df["price_nok_total"], errors="coerce")
    df["carbon_kg_total"] = pd.to_numeric(df["carbon_kg_total"], errors="coerce")

    # Null checks
    n_price  = int(df["price_nok_total"].isna().sum())
    n_carbon = int(df["carbon_kg_total"].isna().sum())
    rows_any = int(df[["price_nok_total","carbon_kg_total"]].isna().any(axis=1).sum())

    print(f"Rows: {len(df):,}")
    print(f"Nulls -> price_nok_total: {n_price:,} | carbon_kg_total: {n_carbon:,}")
    print(f"Rows with any null in target columns: {rows_any:,}")

    # Top items by product_name (sum across rows)
    grp = df.groupby("product_name", dropna=False)[["price_nok_total","carbon_kg_total"]].sum(min_count=1).fillna(0)

    if not grp.empty:
        top_nok_name = grp["price_nok_total"].idxmax()
        top_nok_val  = grp.loc[top_nok_name, "price_nok_total"]
        top_co2_name = grp["carbon_kg_total"].idxmax()
        top_co2_val  = grp.loc[top_co2_name, "carbon_kg_total"]

        print(f"\033[91mMost expensive in NOK   : {top_nok_name!r} -> {top_nok_val}\033[0m")
        print(f"\033[91mHighest CO2 total (kg)  : {top_co2_name!r} -> {top_co2_val}\033[0m")

    else:
        print("Grouped data is empty; cannot compute top items.")

if __name__ == '__main__':
    main()
