import math

import pytest

from haulproof.stats.readiness import (
    miles_to_demonstrate,
    poisson_rate_ci,
    prob_better_than_baseline,
    rate_test_vs_baseline,
)

RAND_FATALITY_RATE = 1.09e-8  # fatalities per mile, U.S. 2013 (Kalra & Paddock, 2016)


def test_miles_to_demonstrate_reproduces_rand():
    # RAND: ~275 million failure-free miles to show < 1.09 fatalities / 100M mi at 95%.
    miles = miles_to_demonstrate(RAND_FATALITY_RATE, confidence=0.95)
    assert miles == pytest.approx(275e6, rel=0.01)


def test_miles_to_demonstrate_grows_with_allowed_events():
    closed = miles_to_demonstrate(1e-6, allowed_events=0)
    assert closed == pytest.approx(-math.log(0.05) / 1e-6)
    # Exact Poisson: P(X <= 2 | mu) = 0.05 at mu ~= 6.296.
    assert miles_to_demonstrate(1e-6, allowed_events=2) == pytest.approx(6.296e6, rel=1e-3)
    # Requiring a 50% improvement over baseline doubles the miles needed.
    assert miles_to_demonstrate(1e-6, improvement=0.5) == pytest.approx(2 * closed)


def test_poisson_ci_known_values():
    # Exact 95% CI for 0 events is [0, 3.689]; for 10 events is [4.795, 18.390].
    _, lo, hi = poisson_rate_ci(0, 1.0)
    assert lo == 0.0
    assert hi == pytest.approx(3.689, abs=1e-3)
    _, lo, hi = poisson_rate_ci(10, 1.0)
    assert lo == pytest.approx(4.795, abs=1e-3)
    assert hi == pytest.approx(18.390, abs=1e-3)


def test_poisson_ci_scales_with_exposure():
    est, lo, hi = poisson_rate_ci(10, 1e6)
    assert est == pytest.approx(1e-5)
    assert lo < est < hi


def test_rate_test_detects_clearly_safer_fleet():
    # Baseline expects 20 events; fleet saw 5.
    result = rate_test_vs_baseline(events=5, exposure=20e6, baseline_rate=1e-6)
    assert result.expected_events == pytest.approx(20)
    assert result.significant
    assert result.rate_ratio == pytest.approx(0.25)


def test_rate_test_not_significant_when_at_baseline():
    result = rate_test_vs_baseline(events=20, exposure=20e6, baseline_rate=1e-6)
    assert not result.significant


def test_bayesian_probability_increases_with_clean_miles():
    p_small = prob_better_than_baseline(0, 1e6, baseline_rate=1e-6)
    p_large = prob_better_than_baseline(0, 1e7, baseline_rate=1e-6)
    assert 0 < p_small < p_large < 1


@pytest.mark.parametrize("events,exposure", [(-1, 1.0), (1.5, 1.0), (1, 0.0)])
def test_invalid_inputs_raise(events, exposure):
    with pytest.raises(ValueError):
        poisson_rate_ci(events, exposure)
