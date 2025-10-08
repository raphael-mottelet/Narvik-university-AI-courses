# dropcol.py  (CSV/TSV autodetect + chunked + column-pruning with drop stats)

import argparse
import csv
import os
import sys
from typing import List
import pandas as pd

#here i keep my columns
PRODUCT_META_COLS: List[str] = [
    "product_name",
    "food_groups",
]

COMMON_NUTRIENT_COLS: List[str] = [
    "energy-kj_100g", "energy-kcal_100g", "energy_100g",
    "fat_100g", "saturated-fat_100g", "cholesterol_100g",
    "carbohydrates_100g", "sugars_100g",
    "fiber_100g", "proteins_100g",
    "salt_100g", 
    #"sodium_100g", sodium and salt are the same but i keep this one just in case for further test with nutriments
    "cholesterol_100g",
    #"energy-kj_100g",
    
    
    #test to see if i can use that one in order to facilitate carbon emission analysis
    "ingredients_tags",
]

# Renamed from ENVIRONMENT_COLS → OTHER_COLS
OTHER_COLS: List[str] = [
    # "ecoscore_score",
    # "ecoscore_grade",
    # "carbon-footprint_100g", no data yet unfortunately
    "nutriscore_grade",
]

#Because for some reason cholesterol is as hard to loose in a dataset as in real life
EXCLUDE_COLS = {"cholesterol_100g", "energy-kj_100g"} 

DEFAULT_INPUT = os.path.join("..", "data", "openfoodfacts", "en.openfoodfacts.org.products.csv")
DEFAULT_OUTPUT = os.path.join("..", "data", "openfoodfacts", "step1.csv")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Slim an OpenFoodFacts CSV/TSV by keeping selected columns, chunked.")
    p.add_argument("--input", default=DEFAULT_INPUT, help=f"Input CSV/TSV (default: {DEFAULT_INPUT})")
    p.add_argument("--output", default=DEFAULT_OUTPUT, help=f"Output CSV (default: {DEFAULT_OUTPUT})")
    p.add_argument("--chunksize", type=int, default=200_000, help="Rows per chunk for streaming read.")
    p.add_argument("--sep", default="", help="Force a delimiter (e.g., '\\t' or ','). Empty = auto-detect.")
    p.add_argument("--progress-every", type=int, default=1_000_000, help="Print a row counter every N rows.")
    return p.parse_args()


def autodetect_sep(path: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        sample = f.read(1024 * 64)
    try:
        return csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"]).delimiter
    except Exception:
        return "\t"  # OFF is TSV


def wanted_columns(_: bool = False) -> List[str]:
    # keep parameter for compatibility; ignored
    cols = ["created_datetime"] + PRODUCT_META_COLS + COMMON_NUTRIENT_COLS + OTHER_COLS
    if "created_t" not in cols:
        cols.append("created_t")
    # remove excluded cols
    cols = [c for c in cols if c not in EXCLUDE_COLS]
    seen = set()
    return [c for c in cols if not (c in seen or seen.add(c))]


def ensure_created_datetime_chunk(df: pd.DataFrame) -> pd.DataFrame:
    if "created_datetime" in df.columns:
        df["created_datetime"] = df["created_datetime"].astype(str)
        return df
    if "created_t" in df.columns:
        s = pd.to_datetime(df["created_t"], unit="s", errors="coerce", utc=True)
        df["created_datetime"] = s.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        df.drop(columns=["created_t"], inplace=True)
        return df
    df["created_datetime"] = ""
    return df


def write_chunk(out_path: str, df: pd.DataFrame, header: bool) -> None:
    df = ensure_created_datetime_chunk(df)
    first = ["created_datetime"]
    rest = [c for c in df.columns if c != "created_datetime"]
    df = df[first + rest]
    df.to_csv(out_path, index=False, mode=("w" if header else "a"), header=header)


def main():
    args = parse_args()

    if not os.path.isfile(args.input):
        print(f"Input file not found: {args.input}", file=sys.stderr)
        sys.exit(2)

    sep = args.sep if args.sep else autodetect_sep(args.input)
    print(f"Detected separator: {repr(sep)}")

    cols_we_want = wanted_columns(False)

    # Read header to compute stats
    with open(args.input, "r", encoding="utf-8", errors="replace") as f:
        header_line = f.readline().rstrip("\n").rstrip("\r")
    actual_cols = header_line.split(sep)

    keep_present = [c for c in cols_we_want if c in actual_cols]
    dropped_cols = [c for c in actual_cols if c not in keep_present]
    excluded_found = [c for c in EXCLUDE_COLS if c in actual_cols]

    print(f"Input column count     : {len(actual_cols)}")
    print(f"Columns to keep (found): {len(keep_present)}")
    print(f"Columns dropped        : {len(dropped_cols)}")

    usecols_param = keep_present if keep_present else None

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    if os.path.exists(args.output):
        os.remove(args.output)

    total_in = 0
    total_out = 0
    header_written = False

    print(f"Streaming from: {args.input}")
    print(f"Writing to    : {args.output}")

    for chunk in pd.read_csv(
        args.input,
        sep=sep,
        usecols=usecols_param,
        chunksize=args.chunksize,
        low_memory=False,
        dtype=str,
        encoding="utf-8",
        on_bad_lines="skip",
    ):
        total_in += len(chunk)

        for col in cols_we_want:
            if col not in chunk.columns:
                chunk[col] = pd.NA

        ordered_cols = [c for c in cols_we_want if c in chunk.columns]
        chunk = chunk[ordered_cols]

        write_chunk(args.output, chunk, header=(not header_written))
        header_written = True
        total_out += len(chunk)

        if args.progress_every and (total_in // args.progress_every) != ((total_in - len(chunk)) // args.progress_every):
            print(f"  processed rows: {total_in:,}")

    print(f"\nSaved cleaned dataset: {args.output}")
    print(f"Rows written   : {total_out:,}")
    print("Done.")


if __name__ == "__main__":
    main()
