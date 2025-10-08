# normalize_compound_tags.py
# Collapses hyphenated tags to their rightmost token: e.g., "corn-sirup" -> "sirup".
# Also: lowercase, remove language prefixes (en:, fr:, ...), standardize separators,
# keep [a-z0-9 -], naive plural trim, and export mapping + counts.

import os, sys, re, argparse
import pandas as pd

DEF_IN  = os.path.join("..", "data", "openfoodfacts", "foodtags-full.csv")
DEF_MAP = os.path.join("..", "data", "openfoodfacts", "foodtags-normalized-mapping.csv")
DEF_CNT = os.path.join("..", "data", "openfoodfacts", "foodtags-normalized-counts.csv")

LANG_PREFIX = re.compile(r"^[a-z]{2}:", re.I)

def normalize_tag(t: str) -> str:
    t = str(t).strip().lower()
    t = LANG_PREFIX.sub("", t)
    t = re.sub(r"[／_/]+", "-", t)                 # unify separators
    t = re.sub(r"[^a-z0-9\-\s]", "", t)           # keep letters/digits/hyphen/space
    t = re.sub(r"\s+", "-", t).strip("-")         # collapse spaces to hyphen
    if "-" in t:
        t = t.split("-")[-1]                      # keep head word (rightmost)
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        t = t[:-1]                                # naive singularization
    return t

def main():
    ap = argparse.ArgumentParser(description="Normalize hyphenated food tags to head words.")
    ap.add_argument("--input",  default=DEF_IN,  help=f"Input CSV path (default: {DEF_IN})")
    ap.add_argument("--mapout", default=DEF_MAP, help=f"Output mapping CSV (default: {DEF_MAP})")
    ap.add_argument("--countout", default=DEF_CNT, help=f"Output counts CSV (default: {DEF_CNT})")
    ap.add_argument("--col", default=None, help="Column name if file has a header (default: first column)")
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        print(f"Input not found: {args.input}", file=sys.stderr); sys.exit(2)

    # Load with/without header
    try:
        df = pd.read_csv(args.input)
        col = args.col or (args.col if args.col in df.columns else df.columns[0])
    except Exception:
        df = pd.read_csv(args.input, header=None, names=["tag"])
        col = "tag"

    if col not in df.columns:
        print(f"Column '{col}' not found. Available: {list(df.columns)}", file=sys.stderr); sys.exit(2)

    df[col] = df[col].astype(str).str.strip().str.lower()
    df["normalized_tag"] = df[col].apply(normalize_tag)
    df = df[df["normalized_tag"].str.len() > 0].copy()

    counts = df["normalized_tag"].value_counts().reset_index()
    counts.columns = ["normalized_tag", "count"]

    os.makedirs(os.path.dirname(args.mapout) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.countout) or ".", exist_ok=True)

    df[[col, "normalized_tag"]].to_csv(args.mapout, index=False)
    counts.to_csv(args.countout, index=False)

    print(f"Saved mapping : {args.mapout}  (rows: {len(df):,})")
    print(f"Saved counts  : {args.countout} (unique normalized: {len(counts):,})")

if __name__ == "__main__":
    main()
