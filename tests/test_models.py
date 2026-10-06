from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from fraudlens.config import ImbalanceConfig, ModelSearch
from fraudlens.data import Split, xy
from fraudlens.models import (
    BOOSTING,
    FOREST,
    ISOLATION,
    LOGISTIC,
    MODEL_NAMES,
    IsolationForestScorer,
    build_pipeline,
    calibrate,
    fraud_scores,
    search_on_validation,
)

SMALL = {
    LOGISTIC: {"C": 1.0},
    FOREST: {"n_estimators": 20, "max_depth": 6},
    BOOSTING: {"max_iter": 30},
    ISOLATION: {"n_estimators": 30, "max_samples": 256},
}


@pytest.fixture(scope="module")
def slices(fixture_df: pd.DataFrame, fixture_split: Split) -> tuple:  # type: ignore[type-arg]
    return (*xy(fixture_df, fixture_split.train), *xy(fixture_df, fixture_split.validation))


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_each_model_trains_and_returns_probabilities(name: str, slices: tuple) -> None:  # type: ignore[type-arg]
    X_tr, y_tr, X_va, y_va = slices
    pipe = build_pipeline(name, SMALL[name], seed=0).fit(X_tr, y_tr)
    scores = fraud_scores(pipe, X_va)
    assert scores.shape == (len(X_va),)
    assert np.all((scores >= 0) & (scores <= 1))
    # The fixture has real signal; every model should beat the no-skill line.
    assert average_precision_score(y_va, scores) > 2 * y_va.mean()


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_seeds_make_results_repeatable(name: str, slices: tuple) -> None:  # type: ignore[type-arg]
    X_tr, y_tr, X_va, _ = slices
    a = fraud_scores(build_pipeline(name, SMALL[name], seed=3).fit(X_tr, y_tr), X_va)
    b = fraud_scores(build_pipeline(name, SMALL[name], seed=3).fit(X_tr, y_tr), X_va)
    # Parallel tree averaging can reorder float sums (~1e-16); anything larger is a seed bug.
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("sampling", ["class_weight", "undersample", "smote", "none"])
def test_sampling_strategies(sampling: str, slices: tuple) -> None:  # type: ignore[type-arg]
    X_tr, y_tr, X_va, _ = slices
    pipe = build_pipeline(LOGISTIC, {"sampling": sampling}, seed=0, imbalance=ImbalanceConfig())
    pipe.fit(X_tr, y_tr)
    assert ("resample" in pipe.named_steps) == (sampling in {"undersample", "smote"})
    assert fraud_scores(pipe, X_va).shape == (len(X_va),)


def test_invalid_configurations() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        build_pipeline("svm", {}, seed=0)
    with pytest.raises(ValueError, match="sampling strategy"):
        build_pipeline(LOGISTIC, {"sampling": "magic"}, seed=0)
    with pytest.raises(ValueError, match="unsupervised"):
        build_pipeline(ISOLATION, {"sampling": "smote"}, seed=0)


def test_isolation_forest_ignores_fraud_rows() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3))
    y = np.zeros(300, dtype=int)
    y[:30] = 1
    X_shifted = X.copy()
    X_shifted[:30] += 100.0  # moving fraud rows must not change a legit-only fit
    a = IsolationForestScorer(n_estimators=20, max_samples=64, random_state=0).fit(X, y)
    b = IsolationForestScorer(n_estimators=20, max_samples=64, random_state=0).fit(X_shifted, y)
    np.testing.assert_array_equal(a.predict_proba(X), b.predict_proba(X))
    assert set(a.predict(X)) <= {0, 1}


def test_search_uses_budget_and_picks_best(slices: tuple) -> None:  # type: ignore[type-arg]
    X_tr, y_tr, X_va, y_va = slices
    spec = ModelSearch(n_iter=3, space={"sampling": ["class_weight"], "C": [0.001, 0.1, 10.0]})
    result = search_on_validation(LOGISTIC, spec, X_tr, y_tr, X_va, y_va, seed=0)
    assert len(result.trials) == 3
    assert result.best_score == max(t.validation_pr_auc for t in result.trials)
    # A budget larger than the grid is capped at the grid size.
    spec_big = ModelSearch(n_iter=50, space={"C": [0.1, 1.0]})
    assert len(search_on_validation(LOGISTIC, spec_big, X_tr, y_tr, X_va, y_va, 0).trials) == 2


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_calibration_keeps_probabilities_valid(method: str, slices: tuple) -> None:  # type: ignore[type-arg]
    X_tr, y_tr, X_va, y_va = slices
    pipe = build_pipeline(BOOSTING, SMALL[BOOSTING], seed=0).fit(X_tr, y_tr)
    before = fraud_scores(pipe, X_va)
    cal = calibrate(pipe, X_va, y_va, method)
    after = fraud_scores(cal, X_va)
    assert np.all((after >= 0) & (after <= 1))
    # The underlying model is frozen, not refitted.
    np.testing.assert_array_equal(fraud_scores(pipe, X_va), before)
