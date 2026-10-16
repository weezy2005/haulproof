"""Driverless-readiness statistics.

Treats safety-relevant events (crashes, injuries, fatalities) as a Poisson process
in miles driven. Given a fleet's observed events and exposure, these functions
answer:

* What is the plausible range of the fleet's true event rate?   -> poisson_rate_ci
* Is the fleet's rate significantly below a human baseline?     -> rate_test_vs_baseline
* What is the posterior probability the fleet beats baseline?   -> prob_better_than_baseline
* How many miles are needed to demonstrate superiority?          -> miles_to_demonstrate

All rates are events per mile unless stated otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats


def _check_inputs(events: int, exposure: float) -> None:
    if events < 0 or int(events) != events:
        raise ValueError(f"events must be a non-negative integer, got {events!r}")
    if exposure <= 0:
        raise ValueError(f"exposure must be positive, got {exposure!r}")


def poisson_rate_ci(
    events: int, exposure: float, confidence: float = 0.95
) -> tuple[float, float, float]:
    """Exact (Garwood) confidence interval for a Poisson rate.

    Returns (point_estimate, lower, upper) in events per unit exposure.
    The exact interval is used rather than a normal approximation because
    safety-critical event counts are typically small (often zero).
    """
    _check_inputs(events, exposure)
    alpha = 1.0 - confidence
    lower = 0.0 if events == 0 else stats.chi2.ppf(alpha / 2, 2 * events) / 2
    upper = stats.chi2.ppf(1 - alpha / 2, 2 * events + 2) / 2
    return events / exposure, lower / exposure, upper / exposure


@dataclass(frozen=True)
class RateTestResult:
    events: int
    exposure: float
    baseline_rate: float
    expected_events: float
    rate_ratio: float
    p_value: float
    significant: bool


def rate_test_vs_baseline(
    events: int, exposure: float, baseline_rate: float, alpha: float = 0.05
) -> RateTestResult:
    """One-sided exact Poisson test of H0: fleet rate >= baseline_rate.

    A small p-value is evidence the fleet's rate is *below* the baseline.
    The baseline is treated as known; see docs/methodology.md for the
    limitation and the planned two-sample extension.
    """
    _check_inputs(events, exposure)
    if baseline_rate <= 0:
        raise ValueError("baseline_rate must be positive")
    expected = baseline_rate * exposure
    p_value = float(stats.poisson.cdf(events, expected))
    return RateTestResult(
        events=events,
        exposure=exposure,
        baseline_rate=baseline_rate,
        expected_events=expected,
        rate_ratio=(events / exposure) / baseline_rate,
        p_value=p_value,
        significant=p_value < alpha,
    )


def prob_better_than_baseline(
    events: int,
    exposure: float,
    baseline_rate: float,
    prior_shape: float = 0.5,
    prior_rate: float = 0.0,
) -> float:
    """Posterior P(fleet rate < baseline_rate) under a Gamma-Poisson model.

    Default prior is the Jeffreys prior Gamma(0.5, 0). An informative prior
    (e.g. from simulation results) can be supplied as pseudo-events
    (prior_shape) over pseudo-miles (prior_rate).
    """
    _check_inputs(events, exposure)
    shape = prior_shape + events
    rate = prior_rate + exposure
    return float(stats.gamma.cdf(baseline_rate, a=shape, scale=1.0 / rate))


def miles_to_demonstrate(
    baseline_rate: float,
    confidence: float = 0.95,
    allowed_events: int = 0,
    improvement: float = 0.0,
) -> float:
    """Miles needed to show, at `confidence`, that the fleet rate is below target.

    target = baseline_rate * (1 - improvement). Finds the exposure m such that
    observing `allowed_events` or fewer events would reject the hypothesis that
    the true rate is >= target. With allowed_events=0 this reduces to the
    closed form m = -ln(1 - confidence) / target (Kalra & Paddock, RAND 2016).
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    if not 0 <= improvement < 1:
        raise ValueError("improvement must be in [0, 1)")
    target = baseline_rate * (1.0 - improvement)
    alpha = 1.0 - confidence
    if allowed_events == 0:
        return -np.log(alpha) / target
    # Solve P(X <= k | target * m) = alpha for the expected count, then scale.
    lam = optimize.brentq(
        lambda mu: stats.poisson.cdf(allowed_events, mu) - alpha, 1e-9, 1e4
    )
    return lam / target
