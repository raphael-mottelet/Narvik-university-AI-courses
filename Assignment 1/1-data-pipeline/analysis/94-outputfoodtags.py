import os, sys, argparse
import pandas as pd
import numpy as np

#This file basically help me to generate some of the synthetic data by exporting some columns data, that i then merge with other dataset or some LLM generated data

DEF_IN   = os.path.join("..","data","openfoodfacts","step8.csv")
DEF_OUT  = os.path.join("..","data","maps","foodtags-full.csv")
COL      = "ingredients_tags"

def parse_args():
    ap = argparse.ArgumentParser(description="Export TOP-N most frequent ingredients_tags to CSV.")
    ap.add_argument("--input",  default=DEF_IN,  help=f"Input CSV (default: {DEF_IN})")
    ap.add_argument("--output", default=DEF_OUT, help=f"Output CSV path (default: {DEF_OUT})")
    ap.add_argument("--col",    default=COL,     help=f"Column with tags (default: {COL})")
    ap.add_argument("--sep",    default=r"[|,]", help=r"Regex to split multi-valued cells (default: [|,])")
    ap.add_argument("--top",    type=int, default=100000, help="How many top tags to keep (default: 1000)")
    return ap.parse_args()

def main():
    a = parse_args()

    if not os.path.isfile(a.input):
        print(f"Input not found: {a.input}", file=sys.stderr); sys.exit(2)

    df = pd.read_csv(a.input, low_memory=False, dtype=str)
    print(f"  Loaded rows: {len(df):,}  columns: {len(df.columns)}")

    if a.col not in df.columns:
        print(f"Column '{a.col}' not found in {a.input}", file=sys.stderr); sys.exit(2)

    # seems like the regex save the day again (i spent a eternity on that thing)-> drop 'en:' / 'fr:' etc.
    tags = (
        df[a.col]
          .dropna()
          .astype(str)
          .str.split(a.sep)
          .explode()
          .str.strip()
          .replace("", np.nan)
          .dropna()
          .str.lower()
          .str.replace(r"^[a-z]{2}:", "", regex=True)  
    )
    print(f"  Non-empty tag entries after explode: {len(tags):,}")

# print the lovely tag list for furture synth data
    unique_tags = sorted(tags.unique())
    out_df = pd.DataFrame({"tag": unique_tags})
    print(f"  Unique tags total : {len(unique_tags):,}")

    os.makedirs(os.path.dirname(a.output) or ".", exist_ok=True)
    out_df.to_csv(a.output, index=False)
    print(f"Saved tags CSV: {a.output}")

if __name__ == "__main__":
    main()
