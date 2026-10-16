"""Download NHTSA FARS and CRSS national CSV releases.

Usage:
    python -m haulproof.data.download --years 2020 2021 2022
    python -m haulproof.data.download --years 2022 --datasets fars
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import requests
import truststore

# Verify TLS against the OS certificate store rather than certifi's bundle, so downloads
# work behind corporate proxies and antivirus TLS scanners that install their own root CA.
truststore.inject_into_ssl()

RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"

URLS = {
    "fars": "https://static.nhtsa.gov/nhtsa/downloads/FARS/{year}/National/FARS{year}NationalCSV.zip",
    "crss": "https://static.nhtsa.gov/nhtsa/downloads/CRSS/{year}/CRSS{year}CSV.zip",
}


def download(dataset: str, year: int, raw_dir: Path = RAW_DIR, force: bool = False) -> Path:
    """Download and extract one dataset-year. Returns the extraction directory."""
    if dataset not in URLS:
        raise ValueError(f"unknown dataset {dataset!r}; choose from {sorted(URLS)}")
    out_dir = raw_dir / dataset / str(year)
    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        print(f"[skip] {dataset} {year} already present at {out_dir}")
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    url = URLS[dataset].format(year=year)
    zip_path = out_dir / f"{dataset}{year}.zip"
    print(f"[get]  {url}")
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    zip_path.unlink()
    print(f"[done] {dataset} {year} -> {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--years", type=int, nargs="+", required=True)
    parser.add_argument("--datasets", nargs="+", default=sorted(URLS), choices=sorted(URLS))
    parser.add_argument("--force", action="store_true", help="re-download existing data")
    args = parser.parse_args()
    for dataset in args.datasets:
        for year in args.years:
            download(dataset, year, force=args.force)


if __name__ == "__main__":
    main()
