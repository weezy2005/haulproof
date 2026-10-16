import duckdb
import pandas as pd
import pandera.errors
import pytest

from haulproof.data.codes import TABLES
from haulproof.data.ingest import build, load, run
from haulproof.data.validate import table_fields

# Synthetic cases: (case, body type, GVWR_FROM). Case 2 has only a passenger car and
# case 4 only a light "unknown truck type", so neither is a large-truck crash.
VEHICLES = [(1, 66, 18), (1, 14, 11), (2, 14, 11), (3, 79, 18), (4, 79, 12)]


def _raw_rows(dataset, table, year, n):
    fields = [f for f in table_fields(dataset, table) if f.raw]
    return pd.DataFrame({f.name.upper(): [f.lo] * n for f in fields}).assign(YEAR=year)


def write_raw(raw_dir, dataset, year, *, crash_overrides=None, vehicle_overrides=None):
    """Write a minimal NHTSA-style release: nested folder, upper-case headers, BOM."""
    key = {"fars": "ST_CASE", "crss": "CASENUM"}[dataset]
    out = raw_dir / dataset / str(year) / f"{dataset.upper()}{year}CSV"
    out.mkdir(parents=True)

    cases = sorted({c for c, _, _ in VEHICLES})
    crash = _raw_rows(dataset, "crash", year, len(cases))
    crash[key] = [10000 + c for c in cases]
    crash["STATENAME"] = "Alabama"  # undeclared columns are ignored
    for col, values in (crash_overrides or {}).items():
        crash[col] = values

    vehicle = _raw_rows(dataset, "vehicle", year, len(VEHICLES))
    vehicle[key] = [10000 + c for c, _, _ in VEHICLES]
    vehicle["VEH_NO"] = vehicle.groupby(key).cumcount() + 1
    vehicle["BODY_TYP"] = [b for _, b, _ in VEHICLES]
    vehicle["GVWR_FROM"] = [g for _, _, g in VEHICLES]
    vehicle = vehicle.drop(columns="YEAR").rename(columns={"V_CONFIG": "V_Config"})
    for col, values in (vehicle_overrides or {}).items():
        vehicle[col] = values
    if year < 2020:
        vehicle = vehicle.drop(columns=["GVWR_FROM", "GVWR_TO"])

    crash.to_csv(out / "accident.csv", index=False, encoding="utf-8-sig")
    vehicle.to_csv(out / "vehicle.csv", index=False)


@pytest.mark.parametrize("dataset", sorted(TABLES))
def test_keeps_only_large_truck_crashes(tmp_path, dataset):
    write_raw(tmp_path, dataset, 2022)
    tables = build(dataset, 2022, raw_dir=tmp_path)
    key = {"fars": "st_case", "crss": "casenum"}[dataset]

    assert tables.crash[key].tolist() == [10001, 10003]
    assert tables.crash["n_large_trucks"].tolist() == [1, 1]
    assert tables.vehicle["body_typ"].tolist() == [66, 79]
    assert (tables.vehicle["year"] == 2022).all()
    assert "statename" not in tables.crash.columns


def test_sentinels_become_missing(tmp_path):
    write_raw(
        tmp_path,
        "fars",
        2022,
        crash_overrides={"HOUR": [99, 14, 3, 7], "LATITUDE": [77.7777, 33.5, 40.1, 41.0]},
        vehicle_overrides={"TRAV_SP": [998, 60, 50, 997, 45], "MOD_YEAR": 9999},
    )
    tables = build("fars", 2022, raw_dir=tmp_path)

    assert tables.crash["hour"].tolist() == [pd.NA, 3]
    assert tables.crash["hour"].dtype == "Int64"
    assert tables.crash["latitude"].isna().tolist() == [True, False]
    assert tables.vehicle["trav_sp"].isna().all()  # 998 not reported, 997 top-coded
    assert tables.vehicle["mod_year"].isna().all()
    assert tables.vehicle["veh_no"].notna().all()


def test_unmapped_code_fails_validation(tmp_path):
    write_raw(tmp_path, "crss", 2022, crash_overrides={"HOUR": [25, 1, 2, 3]})
    with pytest.raises(pandera.errors.SchemaErrors, match="hour"):
        build("crss", 2022, raw_dir=tmp_path)


def test_missing_weight_fails_validation(tmp_path):
    write_raw(tmp_path, "crss", 2022, crash_overrides={"WEIGHT": [None, 1.0, 1.0, 1.0]})
    with pytest.raises(pandera.errors.SchemaErrors, match="weight"):
        build("crss", 2022, raw_dir=tmp_path)


def test_missing_required_column_raises(tmp_path):
    write_raw(tmp_path, "fars", 2022)
    path = next(tmp_path.rglob("accident.csv"))
    pd.read_csv(path, encoding="utf-8-sig").drop(columns="LGT_COND").to_csv(path, index=False)
    with pytest.raises(KeyError, match="lgt_cond"):
        build("fars", 2022, raw_dir=tmp_path)


def test_pre_2020_has_no_gvwr(tmp_path):
    write_raw(tmp_path, "fars", 2019)
    tables = build("fars", 2019, raw_dir=tmp_path)

    # Without GVWR an "unknown truck type" cannot be classified as large.
    assert tables.vehicle["body_typ"].tolist() == [66]
    assert tables.vehicle["gvwr_from"].isna().all()


def test_load_replaces_year_and_keeps_others(tmp_path):
    raw, db = tmp_path / "raw", tmp_path / "test.duckdb"
    write_raw(raw, "fars", 2021)
    write_raw(raw, "fars", 2022)

    run([2021, 2022], ["fars"], db_path=db, raw_dir=raw)
    with duckdb.connect(str(db)) as con:
        load(con, build("fars", 2022, raw_dir=raw))  # reload one year
        counts = con.sql("SELECT year, count(*) FROM fars_crash GROUP BY 1 ORDER BY 1").fetchall()
        n_vehicles = con.sql("SELECT count(*) FROM fars_vehicle").fetchone()[0]
        hour_type = con.sql("SELECT typeof(hour) FROM fars_crash LIMIT 1").fetchone()[0]

    assert counts == [(2021, 2), (2022, 2)]
    assert n_vehicles == 4
    assert hour_type == "BIGINT"


@pytest.mark.parametrize("dataset", sorted(TABLES))
@pytest.mark.parametrize("table", ["crash", "vehicle"])
def test_sentinels_lie_outside_valid_ranges(dataset, table):
    for f in table_fields(dataset, table):
        for code in f.missing:
            assert not f.lo <= code <= f.hi, f"{f.name}: sentinel {code} is a valid value"
