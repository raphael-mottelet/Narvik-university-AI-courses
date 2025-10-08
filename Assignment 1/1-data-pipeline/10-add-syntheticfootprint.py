import os, sys, argparse
import pandas as pd

DEF_IN   = os.path.join("..", "data", "openfoodfacts", "step9.csv")
DEF_MAP  = os.path.join("..", "data", "00-synthetic-data", "Synthetic_footprint_carbon_kg.csv")
DEF_OUT  = os.path.join("..", "data", "openfoodfacts", "step10.csv")
DEF_COL  = "ingredients_tags"
OUT_COL  = "carbon_kg_total"

WORD_COLS  = ["ingredients_tags", "token", "word", "tag", "ingredient", "name", "foodcomponent"]
VALUE_COLS = ["synthetic_footprint_carbon_kg", "synthetic_carbon_footprint", "carbon_kg", "carbon", "value"]

def find_col(cols, cands):
    lc = {c.lower(): c for c in cols}
    for c in cands:
        if c.lower() in lc: return lc[c.lower()]
    return None

def load_map(path):
    if not os.path.isfile(path): 
        print(f"Map not found: {path}", file=sys.stderr); sys.exit(2)
    m = pd.read_csv(path, dtype=str, keep_default_na=False)
    wcol = find_col(m.columns, WORD_COLS)
    vcol = find_col(m.columns, VALUE_COLS)
    if not wcol or not vcol:
        print(f"Cannot detect word/value columns in {path}. Found: {list(m.columns)}", file=sys.stderr); sys.exit(2)
    m[wcol] = m[wcol].str.strip().str.lower()
    m[vcol] = pd.to_numeric(m[vcol], errors="coerce").fillna(0.0).astype(float)
    return dict(zip(m[wcol], m[vcol]))

def sum_cell(cell, fmap):
    if not isinstance(cell, str):
        cell = "" if pd.isna(cell) else str(cell)
    parts = [p.strip().lower() for p in cell.split(",") if p.strip()]
    return float(sum(fmap.get(p, 0.0) for p in parts))

def main():
    ap = argparse.ArgumentParser(description="Add carbon footprint total from synthetic map.")
    ap.add_argument("--input",  default=DEF_IN)
    ap.add_argument("--map",    default=DEF_MAP)
    ap.add_argument("--output", default=DEF_OUT)
    ap.add_argument("--col",    default=DEF_COL)
    args = ap.parse_args()

    fmap = load_map(args.map)
    if not os.path.isfile(args.input):
        print(f"Input not found: {args.input}", file=sys.stderr); sys.exit(2)
    df = pd.read_csv(args.input, low_memory=False, dtype=str)
    if args.col not in df.columns:
        print(f"Column '{args.col}' not found.", file=sys.stderr); sys.exit(2)

    df[OUT_COL] = df[args.col].apply(lambda s: sum_cell(s, fmap))
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")

if __name__ == "__main__":
    main()
