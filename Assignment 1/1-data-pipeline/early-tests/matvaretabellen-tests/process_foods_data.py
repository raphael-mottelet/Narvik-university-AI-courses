# clean_foods.py
# Goal: produce a quality-checked, unit-harmonized nutrient matrix per 100 g.
# Inputs (defaults can be overridden via argv):
#   data/matvaretabellen/foods_en.json
#   data/matvaretabellen/nutrients_en.json
#   data/matvaretabellen/food_groups_en.json
# Outputs:
#   data/clean/foods_clean_meta.csv
#   data/clean/nutrient_matrix_100g.csv
#   Prints a compact JSON report (rows kept/dropped, key imputations, missing)

import json, sys, math, pathlib, re
from typing import Any, Iterable, Optional, List
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP

# ---------------- cfg & io ----------------
base = pathlib.Path("../../data/matvaretabellen")
FOODS = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else base/"foods_en.json"
NUTS  = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else base/"nutrients_en.json"
GROUP = pathlib.Path(sys.argv[3]) if len(sys.argv) > 3 else base/"food_groups_en.json"

outdir = pathlib.Path("../../data/clean"); outdir.mkdir(parents=True, exist_ok=True)

# ---------------- helpers ----------------
def to_float(x: Any) -> Optional[float]:
    try:
        if x is None: return None
        if isinstance(x, (int, float)): return float(x)
        return float(str(x).strip().replace(",", "."))
    except:
        return None

# Unit conversion to grams
UNIT_TO_G = {"g": 1.0, "mg": 1e-3, "µg": 1e-6, "ug": 1e-6, "μg": 1e-6}
def qty_to_g(q, unit):
    q = to_float(q)
    if q is None: return None
    u = (unit or "").strip()
    u = {"μg": "µg"}.get(u, u)  # normalize micro sign
    if u in UNIT_TO_G: return q * UNIT_TO_G[u]
    # non-mass nutrients (e.g., IU, RE, mg-ATE) -> return as-is
    return q

def coalesce(*vals):
    for v in vals:
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            return v
    return None

# sodium chloride ↔ sodium conversion factors
NACL_TO_NA = 0.393
NA_TO_NACL = 1.0 / NACL_TO_NA

def pick_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def first_existing(d: dict, keys: Iterable[str], default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default

def safe_get(d: dict, paths: Iterable[str]):
    for p in paths:
        cur = d
        ok = True
        for part in p.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok:
            return cur
    return None

# ---------- pretty number formatting (MAX 5 decimals, no sci-notation) ----------
PREC = 5
MIN_RES = 10 ** -PREC  # values with abs(x) < 1e-5 -> 0

def nice_number(x):
    if pd.isna(x):
        return ""
    try:
        fx = float(x)
        if abs(fx) < MIN_RES:
            return "0"
        d = Decimal(str(fx)).quantize(Decimal("0." + "0"*PREC), rounding=ROUND_HALF_UP)
        s = format(d, "f").rstrip("0").rstrip(".")
        return s if s != "" else "0"
    except Exception:
        # Last-resort fallback: fixed 5 decimals then strip
        return ("%.5f" % float(x)).rstrip("0").rstrip(".")

def format_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    num_cols = df2.select_dtypes(include=["number"]).columns
    for c in num_cols:
        df2[c] = df2[c].map(nice_number)
    return df2

# ---------------- load FOODS ----------------
foods_raw = json.load(open(FOODS, encoding="utf-8"))
foods_list = foods_raw.get("foods") if isinstance(foods_raw, dict) else None
foods: List[dict] = foods_list if foods_list is not None else (foods_raw if isinstance(foods_raw, list) else [])
if not isinstance(foods, list):
    raise ValueError("Could not parse foods list from foods_en.json")

meta_df = pd.json_normalize(foods, sep=".")
food_id_col   = pick_col(meta_df, ["foodId", "id", "food.id", "code"])
food_name_col = pick_col(meta_df, ["foodName", "nameEn", "name.en", "englishName", "name"])
food_grp_col  = pick_col(meta_df, ["foodGroupId", "groupId", "foodGroup.id", "foodGroupIdEn", "fg.id"])
uri_col       = pick_col(meta_df, ["uri", "url", "href"])

kcal_qty_col  = pick_col(meta_df, [
    "calories.quantity", "calories.value", "energyKcal.quantity", "energy.kcal", "energy.kcal.quantity",
    "energyKcal", "kcal"
])
kj_qty_col    = pick_col(meta_df, [
    "energy.quantity", "energy.value", "energyKj.quantity", "energy.kj", "energy.kj.quantity", "energyKj", "kJ"
])

rows_meta = []
for f in foods:
    fid   = safe_get(f, ["foodId", "id", "food.id", "code"])
    fname = safe_get(f, ["foodName", "nameEn", "name.en", "englishName", "name"])
    fgrp  = safe_get(f, ["foodGroupId", "groupId", "foodGroup.id", "foodGroupIdEn", "fg.id"])
    furl  = safe_get(f, ["uri", "url", "href"])
    kcal  = safe_get(f, ["calories.quantity", "calories.value", "energyKcal.quantity", "energy.kcal", "energy.kcal.quantity", "energyKcal", "kcal"])
    kj    = safe_get(f, ["energy.quantity", "energy.value", "energyKj.quantity", "energy.kj", "energy.kj.quantity", "energyKj", "kJ"])
    rows_meta.append({
        "foodId": fid,
        "foodName": fname,
        "foodGroupId": fgrp,
        "uri": furl,
        "calories.quantity": kcal,
        "energy.quantity": kj
    })
meta = pd.DataFrame(rows_meta)

# ---------------- load NUTRIENTS (robust) ----------------
nuts_raw = json.load(open(NUTS, encoding="utf-8"))
if isinstance(nuts_raw, dict) and "nutrients" in nuts_raw:
    nuts_raw = nuts_raw["nutrients"]
nut_meta = pd.json_normalize(nuts_raw, sep=".")

nut_id_col   = pick_col(nut_meta, ["nutrientId", "id", "code", "nutrient.id"])
nut_name_col = pick_col(nut_meta, ["nameEn", "englishName", "name.en", "nameEnGb", "name", "nutrientNameEn"])
nut_unit_col = pick_col(nut_meta, ["unit", "unit.name", "unit.en", "unitName", "unit.abbr"])

missing = [("nutrient id", nut_id_col), ("nutrient name", nut_name_col), ("unit", nut_unit_col)]
missing = [label for label, col in missing if col is None]
if missing:
    raise KeyError(f"Could not find columns for: {', '.join(missing)}. Available nutrient columns: {list(nut_meta.columns)}")

nut_keep = (nut_meta[[nut_id_col, nut_name_col, nut_unit_col]]
            .rename(columns={nut_id_col: "nutrientId", nut_name_col: "nutrientName", nut_unit_col: "unit"}))
nut_ids = set(nut_keep["nutrientId"])

# ---------------- load GROUPS (robust) ----------------
group_raw = json.load(open(GROUP, encoding="utf-8"))
if isinstance(group_raw, dict):
    group_candidates = first_existing(group_raw, ["foodGroups", "groups", "data", "items"], None)
    if group_candidates is not None and isinstance(group_candidates, list):
        group_raw = group_candidates
group_meta = pd.json_normalize(group_raw, sep=".")

grp_id_col   = pick_col(group_meta, ["foodGroupId", "id", "groupId", "foodGroup.id"])
grp_name_col = pick_col(group_meta, ["foodGroupName", "nameEn", "name.en", "englishName", "name"])
if grp_id_col is None or grp_name_col is None:
    raise KeyError(f"Could not find group id/name columns. Available group columns: {list(group_meta.columns)}")
group_meta = group_meta.rename(columns={grp_id_col: "foodGroupId", grp_name_col: "foodGroupName"})[
    ["foodGroupId", "foodGroupName"]
]

# ---------------- clean meta ----------------
before = len(meta)
meta = meta.dropna(subset=["foodId", "foodName"])
meta = meta.drop_duplicates(subset=["foodId"])
after_drop_id = len(meta)

def kcal_to_kj(kcal):
    v = to_float(kcal)
    return None if v is None else v * 4.184

meta["kcal"] = meta["calories.quantity"].apply(to_float)
meta["kJ"]   = meta["energy.quantity"].apply(to_float)
meta["kcal"] = meta.apply(lambda r: coalesce(r["kcal"], (r["kJ"] / 4.184) if r["kJ"] is not None else None), axis=1)
meta["kJ"]   = meta.apply(lambda r: coalesce(r["kJ"], kcal_to_kj(r["kcal"])), axis=1)

meta["flag_kcal_outlier"] = meta["kcal"].apply(lambda x: bool(x is not None and (x < 0 or x > 900)))
meta = meta.merge(group_meta, on="foodGroupId", how="left")

# ---------------- explode nutrients ----------------
def iter_constituents(food_item: dict) -> Iterable[dict]:
    const = first_existing(food_item, ["constituents", "nutrients", "components"], [])
    if not isinstance(const, list): return []
    out = []
    for c in const:
        if not isinstance(c, dict): continue
        nid = first_existing(c, ["nutrientId", "id", "nutrient.id", "code"])
        qty = first_existing(c, ["quantity", "value", "amount"])
        unt = first_existing(c, ["unit", "unit.name", "unit.abbr", "unit.en"])
        out.append({"nutrientId": nid, "quantity": qty, "unit": unt})
    return out

rows_const = []
for f in foods:
    fid = first_existing(f, ["foodId", "id", "food.id", "code"])
    if fid is None: 
        continue
    for c in iter_constituents(f):
        nid = c.get("nutrientId")
        if nid is None:
            continue
        q = c.get("quantity")
        u = c.get("unit")
        if nid not in nut_ids:
            continue
        rows_const.append({
            "foodId": fid,
            "nutrientId": nid,
            "quantity": to_float(q),
            "unit": (str(u) if u is not None else "").strip()
        })

const_df = pd.DataFrame(rows_const)
if not const_df.empty:
    const_df = const_df[~const_df["quantity"].isna()]
else:
    raise ValueError("No nutrient rows were parsed from foods. Check 'constituents' or 'nutrients' arrays.")

const_df["quantity_g"] = const_df.apply(lambda r: qty_to_g(r["quantity"], r["unit"]), axis=1)

mass_like = {"g", "mg", "µg", "ug", "μg"}
nut_units = nut_keep.set_index("nutrientId")["unit"].to_dict()

def final_value(nid, q_g, q_raw):
    u = (nut_units.get(nid) or "").strip()
    return q_g if u in mass_like else q_raw

const_df["value"] = const_df.apply(lambda r: final_value(r["nutrientId"], r["quantity_g"], r["quantity"]), axis=1)

const_df = (const_df.sort_values(["foodId", "nutrientId"])
            .dropna(subset=["value"])
            .drop_duplicates(["foodId", "nutrientId"], keep="first"))

# -------------- derive sodium/NaCl when one missing (ID-agnostic) --------------
def find_id_by_name(patterns: Iterable[str]) -> Optional[Any]:
    for _, row in nut_keep.iterrows():
        name = str(row["nutrientName"]).lower()
        for pat in patterns:
            if re.search(pat, name):
                return row["nutrientId"]
    return None

sodium_id = find_id_by_name([r"\bsodium\b", r"\bna\b"])
salt_id   = find_id_by_name([r"\bsalt\b", r"\bsodium chloride\b", r"\bnacl\b"])

na_cols = pd.DataFrame()
if sodium_id or salt_id:
    na_ids = {i for i in [sodium_id, salt_id] if i is not None}
    if na_ids:
        na_cols = const_df[const_df["nutrientId"].isin(na_ids)].pivot(
            index="foodId", columns="nutrientId", values="value"
        ).rename_axis(None, axis=1).reset_index()

if not na_cols.empty and sodium_id and salt_id:
    if sodium_id not in na_cols.columns:
        na_cols[sodium_id] = pd.NA
    if salt_id not in na_cols.columns:
        na_cols[salt_id] = pd.NA

    na_cols["Na_calc"]   = na_cols[sodium_id]
    na_cols["NaCl_calc"] = na_cols[salt_id]

    na_cols.loc[na_cols["Na_calc"].isna() & na_cols["NaCl_calc"].notna(), "Na_calc"]   = na_cols["NaCl_calc"] * NACL_TO_NA
    na_cols.loc[na_cols["NaCl_calc"].isna() & na_cols["Na_calc"].notna(), "NaCl_calc"] = na_cols["Na_calc"] * NA_TO_NACL

    imp_na = na_cols.melt(id_vars="foodId", value_vars=["Na_calc", "NaCl_calc"],
                          var_name="which", value_name="value").dropna()
    imp_na["nutrientId"] = imp_na["which"].map({"Na_calc": sodium_id, "NaCl_calc": salt_id})
    imp_na = imp_na[["foodId", "nutrientId", "value"]]

    key = const_df.set_index(["foodId", "nutrientId"])
    for _, r in imp_na.iterrows():
        k = (r["foodId"], r["nutrientId"])
        if k in key.index and pd.isna(key.loc[k, "value"]):
            key.loc[k, "value"] = r["value"]
        elif k not in key.index:
            key.loc[k, "value"] = r["value"]
    const_df = key.reset_index()

# -------------- choose sugar metric (name-agnostic) --------------
sugar_name_patterns = [
    r"mono.?and.?di",
    r"\bmono-?\s?&?\s?di",
    r"\btotal sugars?\b",
    r"\bsugars?, total\b",
    r"\bsugars?\b",
    r"\bsugan\b",
    r"\bsukker\b"
]

cand_ids: List[Any] = []
for pat in sugar_name_patterns:
    for _, row in nut_keep.iterrows():
        if re.search(pat, str(row["nutrientName"]).lower()):
            cand_ids.append(row["nutrientId"])
seen = set()
cand_ids = [x for x in cand_ids if not (x in seen or seen.add(x))]

def pick_sugar(sub: pd.DataFrame):
    vals = {r["nutrientId"]: r["value"] for _, r in sub.iterrows()}
    for k in cand_ids:
        if k in vals and pd.notna(vals[k]):
            return vals[k]
    return None

if cand_ids:
    sugar = (const_df[const_df["nutrientId"].isin(set(cand_ids))]
             .groupby("foodId")
             .apply(lambda g: pick_sugar(g.drop(columns=["foodId"], errors="ignore")))
             .rename("TotalSugar_g")
             .reset_index())
else:
    sugar = pd.DataFrame({"foodId": [], "TotalSugar_g": []})

# -------------- pivot to wide matrix --------------
wide = const_df.pivot(index="foodId", columns="nutrientId", values="value").reset_index()
if not sugar.empty:
    wide = wide.merge(sugar, on="foodId", how="left")

# -------------- join meta and light QC filters --------------
out = meta.merge(wide, on="foodId", how="left")

for col in ["Protein", "Fett", "Karbo", "Fiber", "Alko"]:
    if col not in out.columns: out[col] = pd.NA

macro_sum = out[["Protein", "Fett", "Karbo", "Fiber", "Alko"]].fillna(0).sum(axis=1)
out["flag_macro_sum_gt_105g"] = macro_sum > 105
out["flag_macro_sum_lt_0"]    = macro_sum < 0

before_clean = len(out)
out_clean = out[~(out["flag_macro_sum_gt_105g"] | out["flag_macro_sum_lt_0"] | out["flag_kcal_outlier"])].copy()
after_clean = len(out_clean)

# -------------- save (pre-format numbers to strings) --------------
meta_cols = ["foodId","foodName","foodGroupId","foodGroupName","kcal","kJ","uri"]
meta_to_save   = out_clean[meta_cols].sort_values("foodId")
matrix_to_save = out_clean

meta_to_save_fmt   = format_for_csv(meta_to_save)
matrix_to_save_fmt = format_for_csv(matrix_to_save)

meta_to_save_fmt.to_csv(outdir / "foods_clean_meta.csv", index=False, encoding="utf-8")
matrix_to_save_fmt.to_csv(outdir / "nutrient_matrix_100g.csv", index=False, encoding="utf-8")

# -------------- report --------------
energy_imputed = int(((meta["calories.quantity"].isna()) | (meta["energy.quantity"].isna())).sum())
sodium_imputed = 0
if not na_cols.empty and sodium_id and salt_id:
    base_na = na_cols.copy()
    base_na["Na_missing"] = base_na.get(sodium_id).isna() if sodium_id in base_na.columns else True
    base_na["NaCl_missing"] = base_na.get(salt_id).isna() if salt_id in base_na.columns else True
    sodium_imputed = int((base_na["Na_missing"] | base_na["NaCl_missing"]).sum())

report = {
    "input_rows": before,
    "rows_after_drop_missing_id_or_name": after_drop_id,
    "rows_after_qc": after_clean,
    "dropped_by_qc": before_clean - after_clean,
    "columns_in_matrix": int(out_clean.shape[1]),
    "missing_counts_meta": {c: int(out_clean[c].isna().sum()) for c in meta_cols},
    "imputations": {
        "energy_imputed_from_other_unit": energy_imputed,
        "sodium_from_nacl_or_reverse": sodium_imputed,
        "sugar_metric_chosen": int(out_clean.get("TotalSugar_g", pd.Series(dtype=float)).notna().sum())
    },
    "qc_flags_remaining_counts": {
        "flag_macro_sum_gt_105g": int(out_clean["flag_macro_sum_gt_105g"].sum()),
        "flag_macro_sum_lt_0": int(out_clean["flag_macro_sum_lt_0"].sum()),
        "flag_kcal_outlier": int(out_clean["flag_kcal_outlier"].sum())
    }
}
print(json.dumps(report, ensure_ascii=False, indent=2))
