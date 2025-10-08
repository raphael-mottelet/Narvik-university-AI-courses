# filter_off_until_2024_downsample.py
# Reduce Open Food Facts TSV size by:
# 1) Keeping ONLY rows dated up to 2024-12-31 (UTC)
# 2) Deterministically downsampling ~4x (keep ~25% of rows) using a stable hash of the product code
# - Manual TSV reader/writer (no pandas CSV parsing) for robustness
# - Auto-detects date column and a product identifier column for hashing
# - Handles .csv and .csv.gz
# - Skips malformed lines (wrong column count or unparsable date)
# - Atomically replaces the original with the filtered+downsampled file
#
# Usage:
#   python filter_off_until_2024_downsample.py [optional_path_to_csv_or_csv.gz] [optional_factor]
#     optional_factor: integer >= 2 (default 4). 4 means ~4x fewer rows (keep ~25%).
#
# Default path:
#   ../data/openfoodfacts/en.openfoodfacts.org.products.csv

import os
import sys
import gzip
import tempfile
from typing import List, Optional
from datetime import datetime, timezone
import hashlib

# --------- CONFIG ---------
DEFAULT_PATH = os.path.abspath("../data/openfoodfacts/en.openfoodfacts.org.products.csv")
SEP = "\t"

# Priority order for typical OFF date fields
DATE_CANDIDATES = [
    "last_modified_t", "last_modified_datetime",
    "last_updated_t", "last_updated_datetime",
    "created_t", "created_datetime",
]

# Priority order for a stable product identifier column
CODE_CANDIDATES = ["code", "_id", "id", "product_code", "barcode"]

# Keep rows whose date <= cutoff (remove anything past 2024)
CUTOFF_DT = datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
CUTOFF_TS = int(CUTOFF_DT.timestamp())  # for *_t unix-second columns


def _open_text(path: str, mode: str):
    """Open text file with gzip support and UTF-8 encoding."""
    if path.lower().endswith(".gz"):
        return gzip.open(path, mode + "t", encoding="utf-8", newline="")
    return open(path, mode, encoding="utf-8", newline="")


def _split_tsv(line: str) -> List[str]:
    """
    Split by tab; OFF exports are TSV without embedded tabs in fields.
    We purposely ignore quotes to avoid csv parsing issues.
    """
    if line.endswith("\n"):
        line = line[:-1]
    if line.endswith("\r"):
        line = line[:-1]
    return line.split(SEP)


def _detect_index(header_cols: List[str], candidates: List[str]) -> Optional[int]:
    """Return index of the first matching column name by priority, else None."""
    name_to_idx = {c: i for i, c in enumerate(header_cols)}
    for c in candidates:
        if c in name_to_idx:
            return name_to_idx[c]
    return None


def _parse_iso_to_unix(s: str) -> Optional[int]:
    """Parse ISO8601 like '2024-08-02T13:47:35Z' to unix seconds."""
    if not s:
        return None
    try:
        if s.endswith("Z"):
            dt = datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
    except Exception:
        pass
    try:
        s2 = s.replace(" ", "T")
        if s2.endswith("Z"):
            dt = datetime.fromisoformat(s2.replace("Z", "+00:00"))
        elif "+" in s2[10:] or "-" in s2[10:]:
            dt = datetime.fromisoformat(s2)
        else:
            dt = datetime.fromisoformat(s2).replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except Exception:
        return None


def _to_unix(date_str: str, is_unix: bool) -> Optional[int]:
    """Convert date string to unix seconds depending on column type."""
    if is_unix:
        try:
            return int(float(date_str))
        except Exception:
            return None
    return _parse_iso_to_unix(date_str)


def _hash_keep(s: str, factor: int) -> bool:
    """
    Deterministic keep/drop based on md5 hash of a string.
    Keeps rows where int(md5)%factor == 0 -> ~1/factor of rows.
    If s is empty, we fallback to drop 3/4 by using a constant salt to still keep ~1/factor.
    """
    if not s:
        s = "OFF_EMPTY_CODE"
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    val = int(h[:8], 16)  # use first 32 bits
    return (val % factor) == 0


def stream_filter_and_downsample(in_path: str, factor: int) -> tuple[int, int, int]:
    """
    Stream through the file:
      1) Filter rows with date <= CUTOFF
      2) Downsample deterministically by 'factor' (~1/factor kept)
    Replace the input file atomically.
    Returns (rows_read, rows_date_kept, rows_final_kept).
    """
    if not os.path.exists(in_path):
        raise SystemExit(f"Not found: {in_path}")
    if factor < 2:
        raise SystemExit("factor must be >= 2")

    # Temp output next to source; preserve extension (.csv or .csv.gz)
    fd, tmp_path = tempfile.mkstemp(
        suffix=os.path.splitext(in_path)[1],
        dir=os.path.dirname(in_path)
    )
    os.close(fd)

    rows_in = rows_date_kept = rows_final = 0
    wrote_header = False

    with _open_text(in_path, "r") as r, _open_text(tmp_path, "w") as w:
        header_line = r.readline()
        if not header_line:
            pass
        else:
            header_cols = _split_tsv(header_line)
            ncols = len(header_cols)

            date_idx = _detect_index(header_cols, DATE_CANDIDATES)
            code_idx = _detect_index(header_cols, CODE_CANDIDATES)

            # If no date column, just write header and exit (cannot filter by date)
            if date_idx is None:
                w.write(header_line)
                wrote_header = True
            else:
                w.write(header_line)
                wrote_header = True

                date_col_name = header_cols[date_idx]
                is_unix = date_col_name.endswith("_t")

                for line in r:
                    rows_in += 1
                    parts = _split_tsv(line)
                    if len(parts) != ncols:
                        continue  # malformed row

                    ts = _to_unix(parts[date_idx], is_unix)
                    if ts is None or ts > CUTOFF_TS:
                        continue  # drop rows past cutoff or unparsable

                    rows_date_kept += 1

                    # Downsample: prefer stable product code; if none, use entire line
                    key = parts[code_idx] if code_idx is not None else line
                    if _hash_keep(key, factor):
                        w.write(line)
                        rows_final += 1

    # Atomic replace with backup
    bak = in_path + ".bak"
    try:
        if os.path.exists(bak):
            os.remove(bak)
        os.replace(in_path, bak)
        os.replace(tmp_path, in_path)
        os.remove(bak)
    except Exception:
        try:
            if os.path.exists(bak):
                os.replace(bak, in_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        raise

    return rows_in, rows_date_kept, rows_final


def main():
    path = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    factor = int(sys.argv[2]) if len(sys.argv) > 2 else 4  # default 4x less
    rows_in, rows_date_kept, rows_final = stream_filter_and_downsample(path, factor)
    print(f"Input data rows (excluding header): {rows_in}")
    print(f"Rows <= {CUTOFF_DT.date()}: {rows_date_kept}")
    print(f"Kept after ~1/{factor} downsample: {rows_final}")
    print(f"Replaced original with filtered dataset: {path}")


if __name__ == "__main__":
    main()
