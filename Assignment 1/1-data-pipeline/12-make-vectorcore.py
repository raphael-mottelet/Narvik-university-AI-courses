import argparse, os, sys
import numpy as np
import pandas as pd

DEF_IN  = os.path.join("..", "data", "openfoodfacts", "step11-products.csv")
DEF_OUT = os.path.join("..", "data", "openfoodfacts", "step12-vector-core.csv")

# helpers
def qclip01(s: pd.Series, qlo=0.01, qhi=0.99) -> pd.Series:


    s = pd.to_numeric(s, errors="coerce")
    lo, hi = s.quantile(qlo), s.quantile(qhi)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return pd.Series(np.nan, index=s.index)
    if hi <= lo:
        # fallback to min-max over finite values
        finite = s.replace([np.inf, -np.inf], np.nan).dropna()
        if finite.empty:
            return pd.Series(np.nan, index=s.index)
        a, b = finite.min(), finite.max()
        if b <= a:
            # all identical -> return zeros
            return pd.Series(0.0, index=s.index)
        return ((s - a) / (b - a)).clip(0, 1)
    s = s.clip(lower=lo, upper=hi)
    return ((s - lo) / (hi - lo)).clip(0, 1)


def gaussian_closeness(x: pd.Series, target: float, width: float, lo=None, hi=None) -> pd.Series:
    """Exp(-((x-target)/width)^2) in [0..1]; ignores missing instead of filling with zeros."""
    x = pd.to_numeric(x, errors="coerce")
    if lo is not None:
        x = x.clip(lower=lo)
    if hi is not None:
        x = x.clip(upper=hi)
    z2 = ((x - target) / max(width, 1e-9)) ** 2
    return np.exp(-z2)


def nutriscore_text_to_health_score(s: pd.Series) -> pd.Series:
    # Map text grade to ordinal health score: A=1 .. E=5 (i had some strugle with float values being geenrated)
    mapping = {"a":1, "b":2, "c":3, "d":4, "e":5}
    s_txt = s.astype(str).str.strip().str.lower()
    hs = s_txt.map(mapping)
    hs = hs.where(hs.notna(), 3)  # neutral for unknown/not-applicable
    # Ensure 1/5 integers
    return hs.round().clip(1,5).astype(int)


def safe_mean_rowwise(df_sub: pd.DataFrame, fallback=0.5) -> pd.Series:
    
    # Mean across columns, skipping NaNs; if all NaN -> fallback
    m = df_sub.mean(axis=1, skipna=True)
    return m.where(m.notna(), fallback).clip(0,1)


# core build

def build_core(df: pd.DataFrame) -> pd.DataFrame:
    # Taste balance from 4 nutrient 'sweet spots'
    sugar_c = gaussian_closeness(df["sugars_100g"], target=10.0, width=15.0, lo=0, hi=100)
    fat_c   = gaussian_closeness(df["fat_100g"],    target=10.0, width=15.0, lo=0, hi=100)
    salt_c  = gaussian_closeness(df["salt_100g"],   target=0.4,  width=0.5,  lo=0, hi=5)
    fiber_c = gaussian_closeness(df["fiber_100g"],  target=3.0,  width=2.0,  lo=0, hi=30)

    taste_balance = safe_mean_rowwise(
        pd.concat([sugar_c, fat_c, salt_c, fiber_c], axis=1), fallback=0.5
    )

    # Price normalization (lower -> better in GA via (1-price))
    price_norm = qclip01(df.get("price_nok_total", pd.Series(np.nan, index=df.index)), 0.01, 0.99)

    # Health score from Nutri-Score text
    health_score = nutriscore_text_to_health_score(df.get("nutriscore_grade", pd.Series(index=df.index)))

    # Optional carbon
    carbon_norm = None
    if "carbon_kg_total" in df.columns:
        carbon_norm = qclip01(df["carbon_kg_total"], 0.01, 0.99)

    out = pd.DataFrame({
        "taste_balance": taste_balance,
        "price_norm":    price_norm,
        "health_score":  health_score
    })

    if carbon_norm is not None:
        out["carbon_norm"] = carbon_norm

    # Keep a couple of refs for debugging/joins
    for keep in ("product_name", "food_groups", "created_datetime"):
        if keep in df.columns:
            out[keep] = df[keep]

    # Final tidy
    out = out.dropna(subset=["taste_balance", "price_norm"]).reset_index(drop=True)
    return out


# Automated extension when the program is run

def main():
    ap = argparse.ArgumentParser(description="Build core GA vectors from raw product table.")
    ap.add_argument("--input",  default=DEF_IN,  help="Raw products CSV")
    ap.add_argument("--output", default=DEF_OUT, help="Output CSV (core vectors)")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    core = build_core(df)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    core.to_csv(args.output, index=False)

    # Small echo
    print(f"[vector-core] rows={len(core)}  saved -> {os.path.abspath(args.output)}")
    print(core.head(3).to_string(index=False))

if __name__ == "__main__":
    main()
