import argparse
import csv
import os
import sys
from typing import List

import numpy as np
import pandas as pd

DEFAULT_INPUT  = os.path.join("..", "data", "openfoodfacts", "step1.csv")
DEFAULT_OUTPUT = os.path.join("..", "data", "openfoodfacts", "step2.csv")

DEFAULT_REQUIRED: List[str] = ["food_groups","nutriscore_grade","product_name","ingredients_tags"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter rows by required column presence, keep full rows by default.")
    p.add_argument("--input",  default=DEFAULT_INPUT,  help=f"Input CSV/TSV. Default: {DEFAULT_INPUT}")
    p.add_argument("--output", default=DEFAULT_OUTPUT, help=f"Output CSV. Default: {DEFAULT_OUTPUT}")
    p.add_argument("--required", default=",".join(DEFAULT_REQUIRED),
                   help="Comma-separated list of required columns.")
    p.add_argument("--chunksize", type=int, default=200_000, help="Rows per chunk (default 200k).")
    p.add_argument("--sep", default="", help="Force a delimiter (e.g., '\\t' or ','). Empty = auto-detect.")
    p.add_argument("--keep-only-required", dest="keep_only_required", action="store_true",
                   help="Write only the required columns instead of all columns.")
    p.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="Report stats only; do not write output.")
    return p.parse_args()


def autodetect_sep(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        sample = f.read(1024 * 64)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
        return dialect.delimiter
    except Exception:
        return "\t"  # OFF dump is TSV


def normalize_missing_inplace(df: pd.DataFrame, cols: List[str]) -> None:
    """Only touch the columns we use for filtering."""
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
            continue
        s = df[c]
        is_na = s.isna()
        s = s.astype(str).str.strip()
        s = s.where(s != "", np.nan)
        s = s.mask(s.str.lower() == "nan", np.nan)
        s = s.mask(s.isin(["[]", "{}"]), np.nan)
        s[is_na] = np.nan
        df[c] = s


def filter_mask(df: pd.DataFrame, required_cols: List[str]) -> pd.Series:
    present = [c for c in required_cols if c in df.columns]
    if not present:
        return pd.Series(False, index=df.index)
    return df[present].notna().all(axis=1)


def main():
    args = parse_args()
    in_path, out_path = args.input, args.output
    required_cols = [c.strip() for c in args.required.split(",") if c.strip()]

    if not os.path.isfile(in_path):
        print(f"Input file not found: {in_path}", file=sys.stderr)
        sys.exit(2)

    sep = args.sep if args.sep else autodetect_sep(in_path)
    print(f"Detected separator: {repr(sep)}")
    print(f"Required columns: {required_cols}")
    print("Output mode   :", "required-only" if args.keep_only_required else "all columns")

    # discover header to warn about missing requireds
    with open(in_path, "r", encoding="utf-8", errors="replace") as f:
        header_line = f.readline().rstrip("\n").rstrip("\r")
    actual_cols = header_line.split(sep)
    missing_required = [c for c in required_cols if c not in actual_cols]
    if missing_required:
        print("WARNING: These required columns are not in the file and will be treated as missing:")
        for c in missing_required:
            print("  -", c)

    total_in = 0
    total_kept = 0
    header_written = False

    if not args.dry_run:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        if os.path.exists(out_path):
            os.remove(out_path)

    print(f"Streaming from: {in_path}")
    if not args.dry_run:
        print(f"Writing to    : {out_path}")
    else:
        print("DRY-RUN: no output will be written.")

    # Read everey columns so output keeps the full rows
    for chunk in pd.read_csv(
        in_path,
        sep=sep,
        usecols=None,               # keep all columns in each chunk
        chunksize=args.chunksize,
        low_memory=False,
        dtype=str,
        encoding="utf-8",
        on_bad_lines="skip",
    ):
        total_in += len(chunk)

        # normalize only the required cols for filtering
        normalize_missing_inplace(chunk, required_cols)

        # filter mask
        mask = filter_mask(chunk, required_cols)
        kept = chunk.loc[mask]

        total_kept += len(kept)

        if not args.dry_run and len(kept):
            if args.keep_only_required:
                cols_to_write = [c for c in required_cols if c in kept.columns]
                kept_out = kept[cols_to_write].copy()
            else:
                kept_out = kept

            kept_out.to_csv(
                out_path,
                index=False,
                mode=("w" if not header_written else "a"),
                header=(not header_written),
            )
            header_written = True

        if total_in % (args.chunksize * 5) == 0:
            print(f"Processed {total_in:,} rows...")

    dropped = total_in - total_kept
    pct = (dropped / total_in * 100) if total_in else 0.0
    print(f"Rows read   : {total_in:,}")
    print(f"Rows kept   : {total_kept:,}")
    print(f"Rows dropped: {dropped:,} ({pct:.1f}%)")
    if not args.dry_run:
        print(f"Saved: {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
