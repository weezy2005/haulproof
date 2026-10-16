"""FARS/CRSS field definitions: which columns we keep, their sentinel codes, and valid ranges.

This module is the single source of truth for the ingestion pipeline. Each `Field` says
which raw codes mean "unknown / not reported" (mapped to a real missing value) and what
range a cleaned value must fall in (enforced by the Pandera schemas in `validate.py`).

Codes follow the NHTSA FARS/CRSS Analytical User's Manuals and were checked against the
2022 national CSV releases. Coding differs between years; `Field.missing_by_year`
holds per-year overrides, and columns introduced after `Field.since` are filled with NA
for earlier years instead of failing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# BODY_TYP codes for medium/heavy trucks (GVWR > 10,000 lb), following NHTSA's
# "large truck" definition: step vans, single-unit straight trucks, truck-tractors,
# medium/heavy pickups and unknown medium/heavy trucks. Buses (50-59) and motor homes
# (65, 73) are excluded. 71/72 are pre-2020 single-unit/combination codes.
LARGE_TRUCK_BODY_TYPES = frozenset({60, 61, 62, 63, 64, 66, 67, 71, 72, 78})

# BODY_TYP 79 is "unknown truck type (light/medium/heavy)": counted as a large truck
# only when GVWR_FROM places it in Class 3-8 (GVWR_FROM is available from 2020).
UNKNOWN_TRUCK_BODY_TYPE = 79
LARGE_TRUCK_GVWR_CLASSES = frozenset(range(13, 19))


@dataclass(frozen=True)
class Field:
    """One cleaned column. `name` is the lower-cased raw FARS/CRSS variable name."""

    name: str
    lo: float | None = None
    hi: float | None = None
    missing: frozenset[float] = frozenset()
    dtype: str = "Int64"
    nullable: bool = True
    since: int | None = None  # first year the raw variable exists
    raw: bool = True  # False for columns the pipeline adds itself
    missing_by_year: dict[int, frozenset[float]] = field(default_factory=dict)

    def missing_codes(self, year: int) -> frozenset[float]:
        return self.missing_by_year.get(year, self.missing)


def _f(name: str, lo: float, hi: float, *missing: float, **kw) -> Field:
    return Field(name, lo, hi, frozenset(missing), **kw)


def _key(name: str, lo: float = 1, hi: float = 1e7, **kw) -> Field:
    return Field(name, lo, hi, nullable=False, **kw)


NR_UNK = (98, 99)  # "Not Reported", "Reported as Unknown" for two-digit fields
NR_UNK_1 = (8, 9)  # same, for one-digit fields

# --- crash (accident file) -------------------------------------------------------

CRASH_COMMON = [
    _key("year", 1975, 2100),
    _key("month", 1, 12),
    _key("day_week", 1, 7),
    _f("hour", 0, 23, 99),
    _f("minute", 0, 59, 99),
    _f("harm_ev", 1, 93, *NR_UNK),
    _f("man_coll", 0, 11, *NR_UNK),
    _f("reljct2", 1, 20, *NR_UNK),
    _f("typ_int", 1, 11, *NR_UNK),
    _f("rel_road", 1, 12, *NR_UNK),
    _f("wrk_zone", 0, 4),
    _f("lgt_cond", 1, 7, *NR_UNK_1),
    _f("weather", 1, 12, *NR_UNK),
    _key("ve_total", 1, 1000),
    _key("peds", 0, 1000),
    _key("permvit", 0, 10000),
]

FARS_CRASH = [
    _key("state", 1, 56),
    _key("st_case", 10001, 569999),
    *CRASH_COMMON,
    _f("rur_urb", 1, 6, *NR_UNK_1),
    _f("func_sys", 1, 96, *NR_UNK),
    _f("route", 1, 8, 9),
    _f("nhs", 0, 1, 9),
    _f("latitude", 17, 72, 77.7777, 88.8888, 99.9999, dtype="Float64"),
    _f("longitud", -180, -64, 777.7777, 888.8888, 999.9999, dtype="Float64"),
    _key("fatals", 1, 1000),
]

CRSS_CRASH = [
    _key("casenum", 1, 1e12),
    *CRASH_COMMON,
    # survey design: required for national estimates and design-based variance
    _key("psu", 1, 1e6),
    _key("psu_var", 1, 1e6),
    _key("psustrat", 1, 1e6),
    _key("stratum", 1, 1e6),
    _key("region", 1, 4),
    _key("urbanicity", 1, 2),
    _key("weight", 0, 1e6, dtype="Float64"),
    _f("int_hwy", 0, 1, 9),
    _f("max_sev", 0, 8, 9),
    _f("num_inj", 0, 97, *NR_UNK),  # 98 = no person involved, 99 = all injuries unknown
    # NHTSA-imputed versions of fields with missing values (never missing themselves)
    _key("hour_im", 0, 23),
    _key("lgtcon_im", 1, 7),
    _key("weathr_im", 1, 12),
    _key("maxsev_im", 0, 8),
]

# --- vehicle file (large trucks only) --------------------------------------------

VEHICLE_COMMON = [
    _key("veh_no", 1, 1000),
    _key("body_typ", 60, 79),
    _f("gvwr_from", 11, 18, *NR_UNK, since=2020),
    _f("gvwr_to", 11, 18, *NR_UNK, since=2020),
    _f("v_config", 1, 21, 88, 99),  # 88 = qualifying vehicle, configuration unknown
    _f("cargo_bt", 0, 97, *NR_UNK),
    _f("haz_inv", 1, 2),
    _f("trav_sp", 0, 151, 997, 998, 999),  # 997 = top-coded (> 151 mph)
    _f("rollover", 0, 8, 9),
    _f("j_knife", 0, 3, *NR_UNK_1),
    _f("mod_year", 1900, 2100, 9998, 9999),
    _f("vspd_lim", 0, 97, *NR_UNK),
    _f("vsurcond", 0, 11, *NR_UNK),
    _f("vnum_lan", 0, 7, *NR_UNK_1),
    _f("vtrafway", 0, 7, *NR_UNK_1),
    _f("deformed", 0, 7, *NR_UNK_1),
    _f("m_harm", 1, 93, *NR_UNK),
    _f("dr_pres", 0, 1, 9),
    _f("speedrel", 0, 8, 9),
]

FARS_VEHICLE = [
    _key("state", 1, 56),
    _key("st_case", 10001, 569999),
    _key("year", 1975, 2100, raw=False),  # added from the crash file
    *VEHICLE_COMMON,
    _key("deaths", 0, 1000),
]

CRSS_VEHICLE = [
    _key("casenum", 1, 1e12),
    _key("year", 1975, 2100, raw=False),  # added from the crash file
    *VEHICLE_COMMON,
    _key("weight", 0, 1e6, dtype="Float64"),
    _f("max_vsev", 0, 8, 9),
    _key("mxvsev_im", 0, 8),
]

TABLES: dict[str, dict[str, list[Field]]] = {
    "fars": {"crash": FARS_CRASH, "vehicle": FARS_VEHICLE},
    "crss": {"crash": CRSS_CRASH, "vehicle": CRSS_VEHICLE},
}

CASE_KEY = {"fars": "st_case", "crss": "casenum"}

# Columns derived by the pipeline rather than read from the raw files.
DERIVED_CRASH = [_key("n_large_trucks", 1, 1000, raw=False)]
