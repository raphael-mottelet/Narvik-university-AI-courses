import argparse
import os
import sys
import gzip
import shutil
from urllib.parse import urlsplit

import requests

# prorgam that import and extract the data 
DEFAULT_URL = "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"
DEFAULT_OUTDIR = os.path.join("..", "data", "openfoodfacts")


def filename_from_url(url: str) -> str:
    path = urlsplit(url).path
    name = os.path.basename(path)
    if not name:
        name = "download.csv.gz"
    return name


def stream_download(url: str, dest_path: str, chunk_size: int = 1024 * 1024) -> None:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0)) or None
        downloaded = 0
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(f"\rDownloading: {downloaded:,}/{total:,} bytes ({pct:,.1f}%)", end="")
                    else:
                        print(f"\rDownloading: {downloaded:,} bytes", end="")
        print()  # newline after progress


def stream_gzip_extract(gz_path: str, csv_path: str) -> None:
    # Extract gz -> csv by streaming (binary)
    with gzip.open(gz_path, "rb") as gz_file, open(csv_path, "wb") as out_file:
        shutil.copyfileobj(gz_file, out_file, length=1024 * 1024)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download and extract the OpenFoodFacts .csv.gz dump to CSV.")
    p.add_argument("--url", default=DEFAULT_URL, help=f"Source URL (default: {DEFAULT_URL})")
    p.add_argument("--outdir", default=DEFAULT_OUTDIR, help=f"Output directory (default: {DEFAULT_OUTDIR})")
    p.add_argument("--keep-archive", action="store_true", help="Keep the .gz file after extraction (default: delete).")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    try:
        os.makedirs(args.outdir, exist_ok=True)
        archive_name = filename_from_url(args.url)
        gz_path = os.path.join(args.outdir, archive_name)

        print(f"Downloading from: {args.url}")
        print(f"Saving archive to: {gz_path}")
        stream_download(args.url, gz_path)

        # Decide output CSV name
        if archive_name.endswith(".gz"):
            csv_name = archive_name[:-3]
        else:
            csv_name = archive_name.replace(".gz", "").replace(".gzip", "")
            if not csv_name.lower().endswith(".csv"):
                csv_name += ".csv"
        csv_path = os.path.join(args.outdir, csv_name)

        print(f"Extracting to: {csv_path}")
        stream_gzip_extract(gz_path, csv_path)

        if not args.keep_archive:
            try:
                os.remove(gz_path)
                print(f"Deleted archive: {gz_path}")
            except OSError:
                print("Warning: could not delete archive (permission issue?).")

        print("All done.")
        print(f"Data directory: {os.path.abspath(args.outdir)}")
        print(f"CSV file     : {os.path.abspath(csv_path)}")
        return 0

    except requests.RequestException as e:
        print(f"Network error: {e}", file=sys.stderr)
        return 2
    except (OSError, IOError) as e:
        print(f"I/O error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
