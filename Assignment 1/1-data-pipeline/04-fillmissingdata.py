import os, sys, argparse
import pandas as pd
import numpy as np

DEF_IN  = os.path.join("..","data","openfoodfacts","step3.csv")
DEF_OUT = os.path.join("..","data","openfoodfacts","step4.csv")

# Nutriment columns to zero-fill if missing
NUTRI_COLS = [
    "energy-kcal_100g","energy-kj_100g","energy_100g",
    "fat_100g","saturated-fat_100g",
    "carbohydrates_100g","sugars_100g",
    "proteins_100g","fiber_100g",
    "salt_100g","sodium_100g",
]

def parse_args():
    p = argparse.ArgumentParser(description="Fill missing nutriment values with 0.")
    p.add_argument("--input", default=DEF_IN, help="Input CSV")
    p.add_argument("--output", default=DEF_OUT, help="Output CSV")
    p.add_argument("--cols", default=",".join(NUTRI_COLS),
                   help="Comma-separated nutriment columns to fill with 0")
    return p.parse_args()

def main():
    a = parse_args()
    cols = [c.strip() for c in a.cols.split(",") if c.strip()]
    if not os.path.isfile(a.input):
        print(f"Input not found: {a.input}", file=sys.stderr); sys.exit(2)

    df = pd.read_csv(a.input, low_memory=False, dtype=str)
    n_rows = len(df)
    print(f"Loaded {a.input}  rows={n_rows:,}")

    total_created = 0
    total_filled = 0

    # Coerce to numeric and zero-fill (product_name left untouched)
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
            total_created += n_rows
            print(f"[CREATE] {c:<22} -> created column with 0 for all {n_rows:,} rows")
            continue

        # Before stats
        s_num = pd.to_numeric(df[c], errors="coerce")
        n_missing_before = int(s_num.isna().sum())
        n_nonmissing_before = n_rows - n_missing_before

        # Fill
        s_filled = s_num.fillna(0)
        df[c] = s_filled

        # Optional int cast when all values are whole numbers
        as_int = s_filled.dropna().astype(float).apply(float.is_integer).all()
        if as_int:
            df[c] = df[c].astype("Int64")

        # After stats
        n_missing_after = int(pd.isna(df[c]).sum())
        n_filled = n_missing_before - n_missing_after  # should equal n_missing_before
        total_filled += max(n_filled, 0)

        # Print per-column change summary
        min_v = float(pd.to_numeric(df[c], errors="coerce").min()) if n_rows else 0.0
        max_v = float(pd.to_numeric(df[c], errors="coerce").max()) if n_rows else 0.0
        print(
            f"[FILL ] {c:<22} missing: {n_missing_before:,} -> {n_missing_after:,}  "
            f"(filled {max(n_filled,0):,}); non-missing before: {n_nonmissing_before:,}; "
            f"range after: [{min_v:g}, {max_v:g}]"
        )

    os.makedirs(os.path.dirname(a.output) or ".", exist_ok=True)
    df.to_csv(a.output, index=False)
    print(f"\nSaved: {a.output}")
    print(f"Rows               : {n_rows:,}")
    print(f"Total cells created: {total_created:,} (new columns filled with 0)")
    print(f"Total cells filled : {total_filled:,} (NaN → 0)")
    print("Done.")

if __name__ == "__main__":
    main()
