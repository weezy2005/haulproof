"""Pandera schemas for cleaned FARS/CRSS large-truck tables.

Schemas are generated from the `Field` definitions in `codes.py`, so the valid range of
every column is declared once. Because ranges exclude the sentinel codes (e.g. 99 for an
unknown hour), a sentinel that slipped through cleaning fails validation.
"""

from __future__ import annotations

import pandera.pandas as pa

from haulproof.data.codes import (
    CASE_KEY,
    DERIVED_CRASH,
    LARGE_TRUCK_BODY_TYPES,
    LARGE_TRUCK_GVWR_CLASSES,
    TABLES,
    UNKNOWN_TRUCK_BODY_TYPE,
    Field,
)


def _column(f: Field) -> pa.Column:
    checks = []
    if f.lo is not None and f.hi is not None:
        checks.append(pa.Check.in_range(f.lo, f.hi))
    return pa.Column(f.dtype, checks=checks, nullable=f.nullable)


def table_fields(dataset: str, table: str) -> list[Field]:
    fields = TABLES[dataset][table]
    return fields + DERIVED_CRASH if table == "crash" else fields


def schema(dataset: str, table: str) -> pa.DataFrameSchema:
    """Schema for the cleaned `table` ("crash" or "vehicle") of `dataset` ("fars"/"crss")."""
    key = CASE_KEY[dataset]
    columns = {f.name: _column(f) for f in table_fields(dataset, table)}
    checks = []
    if table == "vehicle":
        checks.append(
            pa.Check(
                lambda df: (df["body_typ"] != UNKNOWN_TRUCK_BODY_TYPE)
                | df["gvwr_from"].isin(LARGE_TRUCK_GVWR_CLASSES).fillna(False),
                name="unknown_truck_type_requires_gvwr_class_3_8",
            )
        )
        columns["body_typ"] = pa.Column(
            "Int64",
            pa.Check.isin(sorted(LARGE_TRUCK_BODY_TYPES | {UNKNOWN_TRUCK_BODY_TYPE})),
            nullable=False,
        )
        unique = ["year", key, "veh_no"]
    else:
        unique = ["year", key]
    return pa.DataFrameSchema(
        columns,
        checks=checks,
        unique=unique,
        strict=True,
        ordered=True,
        name=f"{dataset}_{table}",
    )
