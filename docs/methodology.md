# Methodology

## 1. Question

Given an autonomous trucking fleet's operating record (miles driven, safety
events observed, operating conditions), is there sufficient statistical
evidence that the fleet is at least as safe as human truck drivers *operating
under the same conditions*?

## 2. Human baseline

**Numerator — events.** Truck-involved crashes from NHTSA:

- **FARS**: census of fatal crashes. Truck involvement is identified from the
  vehicle file body-type codes (heavy trucks, tractor-trailers).

  A *large truck* is a vehicle with `BODY_TYP` in {60–64, 66, 67, 71, 72, 78}
  (step vans, single-unit trucks, truck-tractors, medium/heavy pickups, unknown
  medium/heavy trucks; GVWR > 10,000 lb). `BODY_TYP` 79 ("unknown truck type") counts
  only when `GVWR_FROM` is Class 3–8, which is available from 2020 onward. Buses and
  motor homes are excluded.
- **CRSS**: probability sample of police-reported crashes. National estimates
  require the survey weights (`WEIGHT`), and variance estimation must respect
  the design (PSU / stratum). Unweighted CRSS counts are *not* national totals.

**Denominator — exposure.** FHWA Highway Statistics vehicle-miles traveled
for combination and single-unit trucks, by road functional class.

**Stratification.** Rates are computed within strata that an AV operational
design domain (ODD) can be mapped onto: road type (interstate vs. other),
lighting, weather, time of day. Comparing an interstate-only AV fleet with an
all-roads human average would be biased.

**Data quality.** FARS/CRSS encode unknowns as sentinel values (e.g. 8/9,
98/99, 998/999 depending on the field) and change coding between years. These
are mapped to explicit missing values, and the share of missingness per
stratum is reported alongside every rate. Sentinel codes and valid ranges are
declared per field in `haulproof/data/codes.py`; the Pandera schemas built from
them reject any value outside the valid range, so an unmapped sentinel fails
ingestion instead of silently entering an average. Codes meaning "not applicable"
(e.g. `ROLLOVER` = 8) are kept as categories; only "not reported" and "unknown"
become missing. Two exceptions keep numeric columns numeric: `TRAV_SP` = 997
(top-coded, > 151 mph) and CRSS `NUM_INJ` = 98 (no person involved) are also missing.

## 3. Event model

Safety events are modeled as a Poisson process in miles. For a fleet with
`k` events in `m` miles:

- **Exact (Garwood) interval** for the rate — appropriate for small `k`.
- **One-sided exact test** of H0: rate ≥ baseline.
- **Bayesian Gamma–Poisson posterior** with a Jeffreys prior by default, or an
  informative prior derived from simulation (pseudo-events over pseudo-miles).
- **Miles-to-demonstrate**: exposure required so that observing ≤ `k` events
  rejects H0 at the chosen confidence. With `k = 0` this is
  `m = −ln(1 − c) / rate`, reproducing Kalra & Paddock (RAND, 2016).

Overdispersion across strata is handled with negative-binomial regression in
the baseline module.

## 4. Severity modeling

Conditional on a collision, predict severity (KABCO injury scale) from crash
conditions using logistic / ordinal regression (interpretable) and gradient
boosting (flexible).

## 5. A/B evaluation of collision models

Models are compared on held-out data with metrics chosen for safety use
(recall on severe outcomes, Brier score, calibration error), not accuracy
alone. Differences are reported with bootstrap confidence intervals, and
paired classifiers are compared with McNemar's test.

## 6. Known limitations

- The baseline rate is treated as known in the single-sample test; its own
  sampling uncertainty (especially CRSS) should be propagated. A two-sample
  rate-ratio test is planned.
- Police-reported crashes undercount minor collisions; AV event logging does
  not. Event definitions must be aligned before comparison.
- Simulated fleet data is a stand-in for real AV logs and is labeled as such.

## References

- Kalra, N., & Paddock, S. M. (2016). *Driving to Safety: How Many Miles of
  Driving Would It Take to Demonstrate Autonomous Vehicle Reliability?* RAND.
- NHTSA. *FARS and CRSS Analytical User's Manuals.*
- FHWA. *Highway Statistics Series*, Table VM-1.
