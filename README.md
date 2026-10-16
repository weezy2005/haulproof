# haulproof

**A statistical framework for driverless-readiness decisions in autonomous trucking.**

> *"Is our driverless truck safe enough to remove the safety driver?"*
> haulproof answers that question quantitatively: it builds an exposure-normalized
> human-driver safety baseline from public U.S. crash data, models collision
> severity, and decides — with stated confidence — whether an AV fleet's record
> beats that baseline and how many more miles it needs to prove it.

---

## Why this exists

Raw crash counts say almost nothing about safety. A road type with more crashes
may simply carry more trucks. To compare an autonomous fleet against human
drivers you need **rates** (events per mile), **matched operating conditions**
(an AV running highway miles at night shouldn't be compared to a human average
that includes urban rush hour), and **honest uncertainty** (zero crashes in
1 million miles is not proof of safety).

haulproof is built around those three ideas.

## Components

| Module | What it does | Status |
|---|---|---|
| `haulproof.data` | Download, clean and validate NHTSA **FARS** (census of fatal crashes) and **CRSS** (weighted national sample of police-reported crashes), filtered to large trucks; join with FHWA vehicle-miles-traveled exposure data; load into DuckDB | ✅ FARS/CRSS ingestion; 🚧 FHWA exposure join |
| `haulproof.baseline` | Human-driver crash rates per 100M miles with confidence intervals, stratified by road type, lighting, weather, time of day; Poisson / negative-binomial rate models | ⏳ planned |
| `haulproof.severity` | Multivariate models (logistic, ordinal, gradient boosting) predicting injury severity given a collision occurred | ⏳ planned |
| `haulproof.abtest` | A/B evaluation of competing collision models: bootstrap CIs on metric deltas, McNemar's test, calibration analysis | ⏳ planned |
| `haulproof.stats.readiness` | The decision engine: exact Poisson rate tests, Bayesian Gamma–Poisson posteriors, and "miles needed to demonstrate" power calculations | ✅ core implemented + tested |
| `dashboard/` | Streamlit app: explore baselines, compare models, run fleet what-if scenarios | ⏳ planned |

## Quick example

```python
from haulproof.stats.readiness import miles_to_demonstrate, poisson_rate_ci

# Human drivers: ~1.09 fatalities per 100M miles (U.S., 2013 — the RAND baseline).
# How many failure-free miles must an AV drive to show it's better, at 95% confidence?
miles_to_demonstrate(baseline_rate=1.09e-8, confidence=0.95)
# -> ~275 million miles   (reproduces Kalra & Paddock, RAND 2016)

# A fleet logged 3 incidents in 12M miles. What's the plausible true rate?
poisson_rate_ci(events=3, exposure=12e6)
# -> exact 95% interval on incidents per mile
```

## Data sources (all public)

- **FARS** — Fatality Analysis Reporting System, NHTSA. Every fatal crash on U.S. public roads.
- **CRSS** — Crash Report Sampling System, NHTSA. Nationally representative, survey-weighted sample of all police-reported crashes.
- **FHWA Highway Statistics** — vehicle-miles traveled by vehicle class and road type (the exposure denominator).

Raw data is downloaded on demand into `data/raw/` and is not committed.

## Getting started

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
pytest

# download one year of FARS + CRSS, filter to large trucks, validate, load into DuckDB
python -m haulproof.data.ingest --years 2022 --download
```

This writes `data/processed/haulproof.duckdb` with four tables — `fars_crash`,
`fars_vehicle`, `crss_crash`, `crss_vehicle` — holding crashes that involved at least
one large truck and the large-truck vehicle records. Unknown / not-reported codes are
stored as SQL `NULL`; CRSS tables keep the survey weight and design variables.

## Methodology

See [`docs/methodology.md`](docs/methodology.md) for the statistical design,
assumptions, and known limitations.

## Roadmap

- [x] Project scaffold, CI, readiness statistics core
- [x] FARS/CRSS ingestion + data-quality validation (Pandera)
- [ ] DuckDB warehouse + SQL views for truck-involved crashes
- [ ] FHWA exposure join → stratified baseline rates
- [ ] Severity models + A/B evaluation harness
- [ ] Simulated AV fleet generator (stand-in for simulator logs)
- [ ] Streamlit dashboard, deployed publicly

## License

MIT
