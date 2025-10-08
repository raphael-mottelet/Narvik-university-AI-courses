import argparse, os
import numpy as np
import pandas as pd
import re

DEF_IN  = os.path.join("..", "data", "openfoodfacts", "step10.csv")
DEF_OUT = os.path.join("..", "data", "openfoodfacts", "step13-vector-full.csv")

def qclip01(s: pd.Series, qlo=0.01, qhi=0.99) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    lo, hi = s.quantile(qlo), s.quantile(qhi)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return pd.Series(np.nan, index=s.index)
    if hi <= lo:
        finite = s.replace([np.inf, -np.inf], np.nan).dropna()
        if finite.empty:
            return pd.Series(np.nan, index=s.index)
        a, b = finite.min(), finite.max()
        if b <= a:
            return pd.Series(0.0, index=s.index)
        return ((s - a) / (b - a)).clip(0, 1)
    s = s.clip(lower=lo, upper=hi)
    return ((s - lo) / (hi - lo)).clip(0, 1)

def gaussian_closeness(x: pd.Series, target: float, width: float, lo=None, hi=None) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    if lo is not None:
        x = x.clip(lower=lo)
    if hi is not None:
        x = x.clip(upper=hi)
    z2 = ((x - target) / max(width, 1e-9)) ** 2
    return np.exp(-z2)

def nutriscore_text_to_health_score(s: pd.Series) -> pd.Series:
    mapping = {"a":1, "b":2, "c":3, "d":4, "e":5}
    s_txt = s.astype(str).str.strip().str.lower()
    hs = s_txt.map(mapping)
    hs = hs.where(hs.notna(), 3)
    return hs.round().clip(1,5).astype(int)

def safe_mean_rowwise(df_sub: pd.DataFrame, fallback=0.5) -> pd.Series:
    m = df_sub.mean(axis=1, skipna=True)
    return m.where(m.notna(), fallback).clip(0,1)

def count_ingredients(tag_str: pd.Series) -> pd.Series:
    s = tag_str.fillna("").astype(str)
    return s.apply(lambda t: len([x for x in re.split(r"[|,]\s*", t.strip()) if x]) )

def has_any(tag_str: pd.Series, keywords) -> pd.Series:
    s = tag_str.fillna("").astype(str).str.lower()
    pat = r"|".join(map(re.escape, keywords))
    return s.str.contains(pat, regex=True).astype(int)

def pick_ingredients_column(df: pd.DataFrame) -> str:
    for c in ["ingredients_tags", "ingredients", "ingredients_text", "ingredients_list", "ingredients_en", "ingredients_tags_en"]:
        if c in df.columns:
            return c
    return ""

def build_full(df: pd.DataFrame) -> pd.DataFrame:
    sugar_c = gaussian_closeness(df.get("sugars_100g"), target=10.0, width=15.0, lo=0, hi=100)
    fat_c   = gaussian_closeness(df.get("fat_100g"),    target=10.0, width=15.0, lo=0, hi=100)
    salt_c  = gaussian_closeness(df.get("salt_100g"),   target=0.4,  width=0.5,  lo=0, hi=5)
    fiber_c = gaussian_closeness(df.get("fiber_100g"),  target=3.0,  width=2.0,  lo=0, hi=30)
    taste_balance = safe_mean_rowwise(pd.concat([sugar_c, fat_c, salt_c, fiber_c], axis=1), 0.5)

    price_norm   = qclip01(df.get("price_nok_total", pd.Series(np.nan, index=df.index)), 0.01, 0.99)
    health_score = nutriscore_text_to_health_score(df.get("nutriscore_grade", pd.Series(index=df.index)))
    carbon_norm  = qclip01(df.get("carbon_kg_total", pd.Series(np.nan, index=df.index)), 0.01, 0.99)

    kcal_norm   = qclip01(df.get("energy-kcal_100g", df.get("energy_100g")), 0.01, 0.99)
    fat_norm    = qclip01(df.get("fat_100g"),           0.01, 0.99)
    satfat_norm = qclip01(df.get("saturated-fat_100g"), 0.01, 0.99)
    carbs_norm  = qclip01(df.get("carbohydrates_100g"), 0.01, 0.99)
    sugars_norm = qclip01(df.get("sugars_100g"),        0.01, 0.99)
    fiber_norm  = qclip01(df.get("fiber_100g"),         0.01, 0.99)
    protein_norm= qclip01(df.get("proteins_100g"),      0.01, 0.99)
    salt_norm   = qclip01(df.get("salt_100g"),          0.01, 0.99)

    ingr_src = pick_ingredients_column(df)
    if ingr_src:
        ingr_raw = df[ingr_src].astype(str)
    else:
        ingr_raw = pd.Series([""] * len(df), index=df.index)

    n_ingredients = count_ingredients(ingr_raw)
    has_nuts       = has_any(ingr_raw, ["nut", "almond", "hazelnut", "peanut", "walnut", "cashew", "pistachio"])
    has_chocolate  = has_any(ingr_raw, ["chocolate", "cocoa"])
    has_seed       = has_any(ingr_raw, ["seed", "sesame", "flax", "chia", "poppy", "sunflower"])
    has_fruit      = has_any(ingr_raw, ["fruit", "apple", "berry", "grape", "raisin", "banana", "apricot", "fig"])

    out = pd.DataFrame({
        "taste_balance": taste_balance,
        "price_norm":    price_norm,
        "health_score":  health_score,
        "carbon_norm":   carbon_norm,
        "kcal_norm":     kcal_norm,
        "fat_norm":      fat_norm,
        "satfat_norm":   satfat_norm,
        "carbs_norm":    carbs_norm,
        "sugars_norm":   sugars_norm,
        "fiber_norm":    fiber_norm,
        "protein_norm":  protein_norm,
        "salt_norm":     salt_norm,
        "n_ingredients": n_ingredients,
        "has_nuts":      has_nuts,
        "has_chocolate": has_chocolate,
        "has_seed":      has_seed,
        "has_fruit":     has_fruit,
        "ingredients_tags": ingr_raw,
    })

    for keep in ("product_name", "food_groups", "created_datetime"):
        if keep in df.columns:
            out[keep] = df[keep]

    out = out.reset_index(drop=True)
    return out

def main():
    df = pd.read_csv(DEF_IN, low_memory=False)
    full = build_full(df)
    os.makedirs(os.path.dirname(os.path.abspath(DEF_OUT)), exist_ok=True)
    full.to_csv(DEF_OUT, index=False)
    print(f"[vector-full] rows={len(full)}  saved -> {os.path.abspath(DEF_OUT)}")
    print(full.head(3).to_string(index=False))

if __name__ == "__main__":
    main()
