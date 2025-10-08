import argparse, os, sys, re
import pandas as pd

DEF_IN  = os.path.join("..","data","openfoodfacts","step5.csv")
DEF_OUT = os.path.join("..","data","openfoodfacts","step6.csv")

SPLIT_RE = re.compile(r"\s*[|,]\s*")  # tags can be '|' or ',' separated
LANG_PREFIX_RE = re.compile(r"^([a-z]{2,3}(?:-[a-z0-9]{2,8})?):", re.IGNORECASE)

def tag_has_too_many_hyphen_words(tag: str) -> bool:

    if not tag:
        return False
    # Remove language prefix if present
    m = LANG_PREFIX_RE.match(tag)
    core = tag[m.end():] if m else tag
    # Split on '-' and count words (ignore empty fragments from double dashes)
    parts = [p for p in core.split("-") if p]
    return len(parts) > 2

def process_series(series: pd.Series) -> tuple[pd.Series, int, int]:

    total_seen = 0
    total_removed = 0
    out_vals = []

    for val in series.fillna(""):
        raw = str(val)
        if not raw.strip():
            out_vals.append(raw)
            continue

        tags = [t.strip() for t in SPLIT_RE.split(raw) if t.strip()]
        total_seen += len(tags)

        keep = []
        for t in tags:
            if tag_has_too_many_hyphen_words(t):
                total_removed += 1
                continue
            keep.append(t)

        out_vals.append(",".join(keep))

    return pd.Series(out_vals, index=series.index), total_seen, total_removed

def main():
    ap = argparse.ArgumentParser(description="Cut ingredient tags with more than two hyphen-separated words.")
    ap.add_argument("--input",  default=DEF_IN,  help=f"Input CSV (default: {DEF_IN})")
    ap.add_argument("--output", default=DEF_OUT, help=f"Output CSV (default: {DEF_OUT})")
    ap.add_argument("--col",    default="ingredients_tags", help="Column to trim (default: ingredients_tags)")
    args = ap.parse_args()

    # Load
    if not os.path.isfile(args.input):
        print(f"Input not found: {args.input}", file=sys.stderr); sys.exit(2)
    df = pd.read_csv(args.input, low_memory=False, dtype=str)

    if args.col not in df.columns:
        print(f"Column '{args.col}' not found. Available: {list(df.columns)}", file=sys.stderr)
        sys.exit(2)

    # Process
    cleaned, seen, removed = process_series(df[args.col])
    changed_rows = (df[args.col].fillna("") != cleaned.fillna("")).sum()

    # Save
    df[args.col] = cleaned
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)

    # Stats
    kept = seen - removed
    print(f"  Tags removed  : {removed:,}")
    print(f"  Tags kept     : {kept:,}")
    print(f"  Rows modified : {changed_rows:,} / {len(df):,}")

if __name__ == "__main__":
    main()
