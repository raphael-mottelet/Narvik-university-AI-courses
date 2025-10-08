import os, sys, argparse
import pandas as pd

DEF_IN      = os.path.join("..", "data", "openfoodfacts", "step8.csv")
DEF_PRICEMAP= os.path.join("..", "data", "00-synthetic-data", "Synthetic_price_nok.csv")
DEF_OUT     = os.path.join("..", "data", "openfoodfacts", "step9.csv")
DEF_COL     = "ingredients_tags"
OUT_COL     = "price_nok_total" 

# At this point i didnt sleep since some time, so just in case my brain goes kaput i made some col candidates to avoid humain errors
WORD_COL_CANDIDATES  = ["ingredients_tags", "token", "word", "tag", "ingredient", "name", "foodcomponent"]
VALUE_COL_CANDIDATES = ["synthetic_price_nok", "price_nok", "price", "value", "price_index"]

def find_col(cols, candidates):
    lc = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lc:
            return lc[cand.lower()]
    return None

def load_price_map(path: str) -> dict:
    if not os.path.isfile(path):
        print(f"Price map not found: {path}", file=sys.stderr); sys.exit(2)
    pm = pd.read_csv(path, dtype=str, keep_default_na=False)
    if pm.empty:
        print(f"Price map is empty: {path}", file=sys.stderr); sys.exit(2)

    word_col  = find_col(pm.columns, WORD_COL_CANDIDATES)
    value_col = find_col(pm.columns, VALUE_COL_CANDIDATES)
    if word_col is None or value_col is None:
        print(f"Could not detect word/value columns in {path}. "
              f"Found columns: {list(pm.columns)}", file=sys.stderr)
        sys.exit(2)

    pm[word_col] = pm[word_col].astype(str).str.strip().str.lower()
    pm[value_col] = pd.to_numeric(pm[value_col], errors="coerce").fillna(0).astype(int)

    # Build dict 
    price_map = dict(zip(pm[word_col], pm[value_col]))

    print(f"Loaded price map: {len(price_map):,} entries "
          f"(word col='{word_col}', value col='{value_col}') from {path}")
    return price_map

def sum_price_for_cell(cell: str, price_map: dict) -> int:
    if not isinstance(cell, str):
        cell = "" if pd.isna(cell) else str(cell)
    if not cell:
        return 0
    parts = [p.strip().lower() for p in cell.split(",") if p.strip()]
    total = 0
    for p in parts:
        total += price_map.get(p, 0)
    return int(total)

def main():
    ap = argparse.ArgumentParser(description="Add NOK price total from synthetic map.")
    ap.add_argument("--input",   default=DEF_IN,       help=f"Input CSV (default: {DEF_IN})")
    ap.add_argument("--pricemap",default=DEF_PRICEMAP, help=f"Synthetic price map CSV (default: {DEF_PRICEMAP})")
    ap.add_argument("--output",  default=DEF_OUT,      help=f"Output CSV (default: {DEF_OUT})")
    ap.add_argument("--col",     default=DEF_COL,      help=f"Tag column name (default: {DEF_COL})")
    args = ap.parse_args()

    # Load price map
    price_map = load_price_map(args.pricemap)

    # Load main data
    if not os.path.isfile(args.input):
        print(f"Input not found: {args.input}", file=sys.stderr); sys.exit(2)
    df = pd.read_csv(args.input, low_memory=False, dtype=str)
    if args.col not in df.columns:
        print(f"Column '{args.col}' not found in input. Available: {list(df.columns)}", file=sys.stderr); sys.exit(2)

    print(f"Loaded main data: {len(df):,} rows from {args.input}")
    print(f"Computing '{OUT_COL}' by summing NOK values for tags in '{args.col}'...")

    # Metrics
    total_rows = len(df)
    all_tags = df[args.col].fillna("").astype(str).str.split(",").explode().str.strip()
    all_tags = all_tags[all_tags.ne("")]
    total_tags = len(all_tags)
    unique_tags = set(all_tags.str.lower().tolist())
    known = sum(1 for t in unique_tags if t in price_map)
    unknown = len(unique_tags) - known

    print(f"Tag stats: {total_tags:,} tags across rows | {len(unique_tags):,} unique")
    print(f" - in price map: {known:,} unique | missing: {unknown:,} unique")

    # Compute totals
    df[OUT_COL] = df[args.col].apply(lambda s: sum_price_for_cell(s, price_map)).astype(int)

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    # Small sanity preview
    try:
        print(df[[args.col, OUT_COL]].head(5).to_string(index=False))
    except Exception:
        pass

if __name__ == "__main__":
    main()
