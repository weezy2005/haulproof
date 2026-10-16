"""Build the large-truck crash tables from raw FARS/CRSS files and load them into DuckDB.

For each dataset-year the pipeline:

1. reads the accident and vehicle files (only the columns declared in `codes.py`),
2. keeps vehicles that are large trucks and the crashes they were involved in,
3. replaces "not reported" / "unknown" sentinel codes with real missing values,
4. validates both tables with the Pandera schemas in `validate.py`,
5. writes them to DuckDB tables `{dataset}_crash` and `{dataset}_vehicle`, replacing
   any rows already loaded for that year so re-runs are idempotent.

Usage:
    python -m haulproof.data.ingest --years 2022
    python -m haulproof.data.ingest --years 2020 2021 2022 --datasets fars --download
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

from haulproof.data.codes import (
    CASE_KEY,
    LARGE_TRUCK_BODY_TYPES,
    LARGE_TRUCK_GVWR_CLASSES,
    TABLES,
    UNKNOWN_TRUCK_BODY_TYPE,
    Field,
)
from haulproof.data.download import RAW_DIR
from haulproof.data.validate import schema, table_fields

DB_PATH = RAW_DIR.parent / "processed" / "haulproof.duckdb"

# NHTSA ships the accident file with a UTF-8 byte-order mark in some years and Latin-1
# text in others; reading as Latin-1 never fails, so strip the mis-decoded BOM instead.
_BOM = ("﻿", "ï»¿")


@dataclass(frozen=True)
class TruckTables:
    dataset: str
    year: int
    crash: pd.DataFrame
    vehicle: pd.DataFrame


def _normalize(column: str) -> str:
    for bom in _BOM:
        column = column.removeprefix(bom)
    return column.strip().lower()


def find_file(dataset_dir: Path, name: str) -> Path:
    """Find `name` anywhere under `dataset_dir`, ignoring case (layouts differ by year)."""
    matches = [p for p in dataset_dir.rglob("*") if p.name.lower() == name.lower()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one {name} under {dataset_dir}, found {len(matches)}"
        )
    return matches[0]


def read_raw(path: Path, fields: list[Field], year: int) -> pd.DataFrame:
    """Read the declared raw columns of one NHTSA CSV, lower-casing column names."""
    wanted = {f.name for f in fields if f.raw}
    df = pd.read_csv(
        path, encoding="latin-1", usecols=lambda c: _normalize(c) in wanted, low_memory=False
    )
    df.columns = [_normalize(c) for c in df.columns]
    for f in fields:
        if not f.raw or f.name in df.columns:
            continue
        if f.since is not None and year < f.since:
            df[f.name] = pd.NA  # variable not collected yet in this year
        else:
            raise KeyError(f"{path.name} ({year}) is missing required column {f.name!r}")
    return df


def is_large_truck(vehicle: pd.DataFrame) -> pd.Series:
    """Large-truck flag from raw (pre-cleaning) BODY_TYP and GVWR_FROM codes."""
    body = vehicle["body_typ"]
    unknown_truck_heavy = (body == UNKNOWN_TRUCK_BODY_TYPE) & vehicle["gvwr_from"].isin(
        LARGE_TRUCK_GVWR_CLASSES
    )
    return (body.isin(LARGE_TRUCK_BODY_TYPES) | unknown_truck_heavy).astype(bool)


def clean(df: pd.DataFrame, fields: list[Field], year: int) -> pd.DataFrame:
    """Map sentinel codes to NA and cast to the declared nullable dtypes, in field order."""
    out = {}
    for f in fields:
        col = df[f.name]
        codes = f.missing_codes(year)
        if codes:
            col = col.mask(col.isin(codes))
        out[f.name] = col.astype(f.dtype)
    return pd.DataFrame(out, index=df.index).reset_index(drop=True)


def build(dataset: str, year: int, raw_dir: Path = RAW_DIR) -> TruckTables:
    """Read, filter, clean and validate one dataset-year."""
    dataset_dir = raw_dir / dataset / str(year)
    key = CASE_KEY[dataset]
    crash_fields = table_fields(dataset, "crash")
    vehicle_fields = TABLES[dataset]["vehicle"]

    vehicle = read_raw(find_file(dataset_dir, "vehicle.csv"), vehicle_fields, year)
    crash = read_raw(find_file(dataset_dir, "accident.csv"), crash_fields, year)

    if (crash["year"] != year).any():
        raise ValueError(f"{dataset} {year}: accident file contains other years")

    trucks = vehicle[is_large_truck(vehicle)].copy()
    trucks["year"] = year
    n_trucks = trucks.groupby(key).size().rename("n_large_trucks")
    crash = crash.merge(n_trucks, left_on=key, right_index=True, how="inner")

    orphans = set(trucks[key]) - set(crash[key])
    if orphans:
        raise ValueError(f"{dataset} {year}: {len(orphans)} truck vehicles have no crash record")

    crash = clean(crash, crash_fields, year).sort_values(key, ignore_index=True)
    trucks = clean(trucks, vehicle_fields, year).sort_values([key, "veh_no"], ignore_index=True)
    return TruckTables(
        dataset=dataset,
        year=year,
        crash=schema(dataset, "crash").validate(crash, lazy=True),
        vehicle=schema(dataset, "vehicle").validate(trucks, lazy=True),
    )


def load(con: duckdb.DuckDBPyConnection, tables: TruckTables) -> None:
    """Write one dataset-year into DuckDB, replacing any previous load of that year."""
    con.begin()
    try:
        for name, df in (("crash", tables.crash), ("vehicle", tables.vehicle)):
            table = f"{tables.dataset}_{name}"
            con.register("_incoming", df)
            con.execute(f"CREATE TABLE IF NOT EXISTS {table} AS SELECT * FROM _incoming LIMIT 0")
            con.execute(f"DELETE FROM {table} WHERE year = ?", [tables.year])
            con.execute(f"INSERT INTO {table} BY NAME SELECT * FROM _incoming")
            con.unregister("_incoming")
        con.commit()
    except Exception:
        con.rollback()
        raise


def run(
    years: list[int],
    datasets: list[str],
    db_path: Path = DB_PATH,
    raw_dir: Path = RAW_DIR,
    download: bool = False,
) -> None:
    if download:
        from haulproof.data.download import download as fetch

        for dataset in datasets:
            for year in years:
                fetch(dataset, year, raw_dir=raw_dir)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(db_path)) as con:
        for dataset in datasets:
            for year in years:
                tables = build(dataset, year, raw_dir)
                load(con, tables)
                print(
                    f"[load] {dataset} {year}: {len(tables.crash):,} crashes, "
                    f"{len(tables.vehicle):,} large trucks -> {db_path}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--years", type=int, nargs="+", required=True)
    parser.add_argument("--datasets", nargs="+", default=sorted(TABLES), choices=sorted(TABLES))
    parser.add_argument("--db", type=Path, default=DB_PATH, help="DuckDB file to write")
    parser.add_argument("--download", action="store_true", help="download missing raw data")
    args = parser.parse_args()
    run(args.years, args.datasets, db_path=args.db, download=args.download)


if __name__ == "__main__":
    main()
