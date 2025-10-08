#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
from pathlib import Path
from typing import Any, Optional, List, Dict

import numpy as np
import pandas as pd


# ---------------- config & args ----------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path",
                    default="../../data/matvaretabellen/foods_en.json",
                    help="Input JSON file (default: ../../data/matvaretabellen/foods_en.json)")
    ap.add_argument("--key", default="foods",
                    help="Top-level key that holds the list. If not found, the script will try to auto-detect.")
    ap.add_argument("--top", type=int, default=10,
                    help="Show top-K columns by missing percentage in stdout (default: 10).")
    ap.add_argument("--out", dest="out_path",
                    default="../data/clean/scan_report.json",
                    help="Path to write JSON report (default: ../../data/clean/scan_report.json). Use '' to skip writing.")
    return ap.parse_args()


# ---------------- helpers ----------------
def is_missing(x: Any) -> bool:
    """Define 'missing' to include None, NaN, '', empty list/dict."""
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    if isinstance(x, str):
        return x.strip() == ""
    if isinstance(x, (list, tuple, set)):
        return len(x) == 0
    if isinstance(x, dict):
        return len(x) == 0
    return False


def find_items_container(raw: Any, preferred_key: Optional[str] = None) -> List[Dict]:
    """
    Return the list of items to scan.
    Strategy:
      1) If raw is a list -> use it.
      2) If raw is a dict and preferred_key exists and is a list -> use it.
      3) Else, if raw is a dict, return the first value that is a list of dicts.
    """
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        if preferred_key and preferred_key in raw and isinstance(raw[preferred_key], list):
            return raw[preferred_key]
        # auto-detect first list-like value that looks like a table
        for v in raw.values():
            if isinstance(v, list) and (len(v) == 0 or isinstance(v[0], (dict, str, int, float, bool))):
                return v
    raise ValueError("Unexpected JSON shape: could not locate a list of items to scan.")


def normalize_foods(foods: List[Dict]) -> pd.DataFrame:
    """Flatten nested objects with dot notation."""
    return pd.json_normalize(foods, sep=".")


def compute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with columns: column, missing_count, missing_pct."""
    total = df.shape[0]
    records = []
    for col in df.columns:
        vals = df[col].tolist()
        miss = sum(is_missing(v) for v in vals)
        records.append({
            "column": col,
            "missing_count": int(miss),
            "missing_pct": (miss / total * 100.0) if total > 0 else 0.0
        })
    out = pd.DataFrame(records).sort_values(["missing_pct", "missing_count", "column"], ascending=[False, False, True])
    return out


# ---------------- main ----------------
def main():
    args = parse_args()
    in_path = Path(args.in_path)
    out_path = Path(args.out_path) if args.out_path else None

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    # load JSON
    with in_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    # find items list
    foods = find_items_container(raw, preferred_key=args.key)
    if not isinstance(foods, list):
        raise ValueError("Items container is not a list.")
    # flatten
    df = normalize_foods(foods)

    # compute missing
    miss_df = compute_missing(df)

    # stdout summary
    print(f"rows: {df.shape[0]}")
    print(f"columns: {df.shape[1]}")
    topk = miss_df.head(max(0, int(args.top)))
    if not topk.empty:
        print("\nTop columns by missing %:")
        for _, r in topk.iterrows():
            print(f"- {r['column']}: {r['missing_count']} missing ({r['missing_pct']:.1f}%)")

    # prepare report
    report = {
        "input_file": str(in_path),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values_per_column": {
            r["column"]: {
                "missing_count": int(r["missing_count"]),
                "missing_pct": round(float(r["missing_pct"]), 3),
            }
            for _, r in miss_df.iterrows()
        }
    }

    # write report JSON if requested
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nSaved scan report to: {out_path}")


if __name__ == "__main__":
    main()
