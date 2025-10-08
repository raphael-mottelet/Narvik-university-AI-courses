import os, sys, re, argparse
import pandas as pd

# This program use a single map word, to reduce word diversity in order to easily attribute prices and carbon footprint later

DEF_IN   = os.path.join("..","data","openfoodfacts","step6.csv")
DEF_OUT  = os.path.join("..","data","openfoodfacts","step7.csv")
DEF_MAP  = os.path.join("..","data","maps","ingredienttags-singleword-map.csv")
DEF_COL  = "ingredients_tags"

#regex save the day
LANG_PREFIX = re.compile(r"^[a-z]{2}:", re.I)
SPLIT_RE    = re.compile(r"[|,]")
NON_ASCII_ALLOWED = re.compile(r"[^a-z\-\s]")   # only a-z, space, hyphen (digits NOT allowed)

def clean_token(t: str) -> str:
    # Baasic cleanup: lowercase, strip lang prefix, drop non-allowed chars, collapse spaces to hyphen
    t = str(t).strip().lower()
    t = LANG_PREFIX.sub("", t)                  # drop 'en:' / 'fr:' ...
    t = re.sub(r"[／_/]+", "-", t)              # unify weird separators to '-'
    t = NON_ASCII_ALLOWED.sub("", t)            # keep only a-z and '-'
    t = re.sub(r"\s+", "-", t).strip("-")
    return t

def collapse_to_single_word(t: str) -> str:

    #If hyphenated -> keep rightmost token; singularize naively.
    if not t:
        return t
    if "-" in t:
        t = t.split("-")[-1]
    # naive singular (no numbers by design, only letters)
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        t = t[:-1]
    return t

def normalize_tag_to_single_word(raw_tag: str) -> str:

    #Full pipeline for a single tag → single word or '' if it becomes invalid/empty.
    t = clean_token(raw_tag)
    if not t:
        return ""
    # If any digit slipped in (shouldn't), drop
    if re.search(r"\d", t):
        return ""
    t = collapse_to_single_word(t)
    # final safety: only a-z
    if not t or re.search(r"[^a-z]", t):
        # allow only letters in the end
        t = re.sub(r"[^a-z]", "", t)
    return t

def normalize_cell(cell: str) -> str:

    # Normalize a whole ingredients_tags cell: split, map each tag, dedup, keep only non-empty.
    if not isinstance(cell, str):
        cell = "" if pd.isna(cell) else str(cell)
    parts = [p.strip() for p in SPLIT_RE.split(cell) if p.strip()]
    out, seen = [], set()
    for p in parts:
        tgt = clean_token(p)
        if not tgt or re.search(r"\d", tgt):
            continue  # drop tags containing numbers entirely
        res = collapse_to_single_word(tgt)
        res = re.sub(r"[^a-z]", "", res)  # final guard: letters only
        if res and res not in seen:
            seen.add(res)
            out.append(res)
    return ",".join(out)

def main():
    ap = argparse.ArgumentParser(description="Build tag map (target,result) and apply it to dataset.")
    ap.add_argument("--input",  default=DEF_IN,  help=f"Input dataset CSV (default: {DEF_IN})")
    ap.add_argument("--output", default=DEF_OUT, help=f"Output normalized dataset CSV (default: {DEF_OUT})")
    ap.add_argument("--mapout", default=DEF_MAP, help=f"Output mapping CSV (default: {DEF_MAP})")
    ap.add_argument("--col",    default=DEF_COL, help=f"Tag column name (default: {DEF_COL})")
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        print(f"Input not found: {args.input}", file=sys.stderr); sys.exit(2)

    df = pd.read_csv(args.input, low_memory=False, dtype=str)
    if args.col not in df.columns:
        print(f"Column '{args.col}' not found. Available: {list(df.columns)}", file=sys.stderr); sys.exit(2)

    # --- Build (target,result) pairs from ALL tags observed
    s = df[args.col].fillna("").astype(str)

    # explode into raw tokens
    tokens = (
        s.str.split(SPLIT_RE)
         .explode()
         .dropna()
         .astype(str)
         .str.strip()
    )

    # clean target (still hyphenated allowed, but no digits, ascii letters + hyphen only)
    targets = tokens.apply(clean_token)
    targets = targets[targets.astype(bool)]
    targets = targets[~targets.str.contains(r"\d")]          # drop if any number
    targets = targets[targets.str.contains(r"[a-z]")]        # must contain a letter

    # result = single word
    results = targets.apply(collapse_to_single_word)
    results = results.str.replace(r"[^a-z]", "", regex=True) # ensure letters only
    valid = results.astype(bool)

    mapping_df = pd.DataFrame({
        "target": targets[valid].values,
        "result": results[valid].values
    }).drop_duplicates()

    # remove trivial empties + ensure both columns non-empty, letters only
    mapping_df = mapping_df[
        mapping_df["target"].str.contains(r"[a-z]") &
        mapping_df["result"].str.contains(r"^[a-z]+$")
    ].drop_duplicates()

    # --- Apply map to dataset (fast path uses normalize_cell directly, which matches the same logic)
    before = df[args.col].fillna("").astype(str)
    after  = before.apply(normalize_cell)
    changed = (before != after).sum()
    df[args.col] = after

    # --- Save outputs
    os.makedirs(os.path.dirname(args.mapout) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    mapping_df.to_csv(args.mapout, index=False)
    df.to_csv(args.output, index=False)

    print(f"Mapping saved : {args.mapout}  (pairs: {len(mapping_df):,})")
    print(f"Dataset saved : {args.output}   (rows: {len(df):,})")
    print(f"Rows changed  : {changed:,}")

if __name__ == "__main__":
    main()
