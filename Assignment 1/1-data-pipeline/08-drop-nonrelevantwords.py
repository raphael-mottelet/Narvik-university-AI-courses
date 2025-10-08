import os, sys, argparse, re
import pandas as pd

DEF_IN      = os.path.join("..", "data", "openfoodfacts", "step7.csv")
DEF_NONFOOD = os.path.join("..", "data","maps", "non_food_single_words.csv")
DEF_OUT     = os.path.join("..", "data", "openfoodfacts", "step8.csv")
DEF_COL     = "ingredients_tags"

#This program remove non food related words that have been introduced or messed up by the synthetic map

WORD_OK = re.compile(r"^[a-z]+$")  # keep only plain a–z tags

def load_nonfood_set(path: str) -> set:
    if not os.path.isfile(path):
        print(f"Non-food list not found: {path}", file=sys.stderr); sys.exit(2)
    nf = pd.read_csv(path, header=None, names=["w"], dtype=str, keep_default_na=False)
    nf["w"] = nf["w"].str.strip().str.lower()
    nf = nf[nf["w"].str.match(WORD_OK, na=False)]
    return set(nf["w"].tolist())

def clean_cell(cell: str, drop_set: set) -> str:

    if not isinstance(cell, str):
        cell = "" if pd.isna(cell) else str(cell)
    if not cell:
        return ""
    parts = [p.strip().lower() for p in cell.split(",") if p.strip()]
    kept = [p for p in parts if (p not in drop_set and WORD_OK.match(p))]
    # dedupe but preserve order
    seen, out = set(), []
    for k in kept:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return ",".join(out)

def main():
    ap = argparse.ArgumentParser(description="Drop non-food tags and rows that become empty.")
    ap.add_argument("--input",  default=DEF_IN,      help=f"Input CSV (default: {DEF_IN})")
    ap.add_argument("--nonfood", default=DEF_NONFOOD, help=f"Non-food words CSV/TXT (default: {DEF_NONFOOD})")
    ap.add_argument("--output", default=DEF_OUT,     help=f"Output CSV (default: {DEF_OUT})")
    ap.add_argument("--col",    default=DEF_COL,     help=f"Tag column name (default: {DEF_COL})")
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        print(f"Input not found: {args.input}", file=sys.stderr); sys.exit(2)

    nonfood = load_nonfood_set(args.nonfood)
    print(f"Loaded non-food list: {len(nonfood):,} words from {args.nonfood}")

    df = pd.read_csv(args.input, low_memory=False, dtype=str)
    if args.col not in df.columns:
        print(f"Column '{args.col}' not found. Available: {list(df.columns)}", file=sys.stderr); sys.exit(2)

    before = df[args.col].fillna("").astype(str)
    total_rows_before = len(df)
    total_tags_before = (
        before.str.split(",").explode().str.strip().replace("", pd.NA).dropna().shape[0]
    )

    # Clean tags
    df[args.col] = before.apply(lambda s: clean_cell(s, nonfood))

    after = df[args.col]
    total_tags_after = (
        after.str.split(",").explode().str.strip().replace("", pd.NA).dropna().shape[0]
    )
    changed_rows = (before != after).sum()

    # Drop rows with empty tags
    empty_mask = after.str.strip().eq("")
    dropped_rows = int(empty_mask.sum())
    df = df.loc[~empty_mask].copy()

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"Rows dropped (empty): {dropped_rows:,}")
    print(f"Tags removed (sum)  : {total_tags_before - total_tags_after:,}")
    
if __name__ == "__main__":
    main()