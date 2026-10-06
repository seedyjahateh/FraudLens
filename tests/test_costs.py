from __future__ import annotations

import itertools

import numpy as np
import pytest

from fraudlens.config import CostConfig
from fraudlens.costs import (
    baseline_costs,
    choose_threshold,
    cost_at_threshold,
    cost_curve,
    missed_costs,
    savings,
)

COSTS = CostConfig(missed_fraud="amount", false_alarm=5.0, caught_fraud=5.0)


def test_known_optimum_on_constructed_example() -> None:
    # Scores in descending order: fraud $100, legit, fraud $3, legit, legit.
    # Flag top 1: 5 (review) + 3 (missed small fraud) = 8
    # Flag top 2: 5 + 5 + 3 = 13; flag top 3: 15; flag nothing: 103; flag all: 25.
    y = np.array([1, 0, 1, 0, 0])
    s = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
    amounts = np.array([100.0, 20.0, 3.0, 50.0, 10.0])
    best = choose_threshold(y, s, amounts, COSTS)
    assert best.threshold == pytest.approx(0.9)
    assert best.total_cost == pytest.approx(8.0)
    assert (best.tp, best.fp, best.fn, best.tn) == (1, 0, 1, 3)


def test_curve_matches_brute_force() -> None:
    rng = np.random.default_rng(0)
    y = (rng.random(200) < 0.1).astype(int)
    s = np.round(rng.random(200), 2)  # rounding forces ties
    amounts = rng.lognormal(3, 1, 200)
    thresholds, totals = cost_curve(y, s, amounts, COSTS)
    for t, total in zip(thresholds, totals, strict=True):
        assert total == pytest.approx(cost_at_threshold(y, s, amounts, t, COSTS).total_cost)
    brute = min(
        cost_at_threshold(y, s, amounts, t, COSTS).total_cost
        for t in itertools.chain(np.unique(s), [2.0])
    )
    assert choose_threshold(y, s, amounts, COSTS).total_cost == pytest.approx(brute)


def test_first_threshold_flags_nothing() -> None:
    y = np.array([1, 0])
    s = np.array([1.0, 0.2])
    thresholds, totals = cost_curve(y, s, np.array([40.0, 1.0]), COSTS)
    assert thresholds[0] > 1.0
    assert totals[0] == pytest.approx(40.0)


def test_baselines_and_savings_arithmetic() -> None:
    y = np.array([1, 1, 0, 0, 0])
    amounts = np.array([100.0, 50.0, 1.0, 1.0, 1.0])
    base = baseline_costs(y, amounts, COSTS)
    assert base["flag_nothing"] == pytest.approx(150.0)
    assert base["flag_everything"] == pytest.approx(25.0)
    saved, pct = savings(30.0, 150.0)
    assert saved == pytest.approx(120.0)
    assert pct == pytest.approx(80.0)
    assert savings(1.0, 0.0) == (-1.0, 0.0)


def test_fixed_missed_fraud_cost() -> None:
    costs = CostConfig(missed_fraud=200.0, false_alarm=1.0, caught_fraud=0.0)
    y = np.array([1, 0, 1])
    np.testing.assert_allclose(missed_costs(y, np.array([5.0, 5.0, 5.0]), costs), [200, 0, 200])


def test_cost_config_validation() -> None:
    with pytest.raises(ValueError):
        CostConfig(missed_fraud="nonsense")
    with pytest.raises(ValueError):
        CostConfig(false_alarm=-1.0)


def test_ties_prefer_fewer_flags() -> None:
    # Flagging the legit row costs 5 and changes nothing else: never worth it.
    y = np.array([0])
    best = choose_threshold(y, np.array([0.4]), np.array([10.0]), COSTS)
    assert best.tp + best.fp == 0
