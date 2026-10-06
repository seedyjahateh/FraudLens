"""Model definitions, imbalance strategies and the validation-slice search (FR-4..FR-9, EV-5).

Every model is an imbalanced-learn ``Pipeline``::

    FeatureBuilder -> StandardScaler -> [sampler] -> estimator

The scaler and any sampler are fitted inside the pipeline on training data only, and
imbalanced-learn applies samplers during ``fit`` but never during ``predict`` — so
validation and test rows are never resampled (EV-5, EV-6).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterSampler
from sklearn.preprocessing import StandardScaler

from fraudlens.config import ImbalanceConfig, ModelSearch
from fraudlens.features import FeatureBuilder

LOGISTIC = "logistic_regression"
FOREST = "random_forest"
BOOSTING = "hist_gradient_boosting"
ISOLATION = "isolation_forest"
MODEL_NAMES: list[str] = [LOGISTIC, FOREST, BOOSTING, ISOLATION]
SUPERVISED: list[str] = [LOGISTIC, FOREST, BOOSTING]
SAMPLING_STRATEGIES = {"class_weight", "undersample", "smote", "none"}


class IsolationForestScorer(ClassifierMixin, BaseEstimator):
    """FR-7: an isolation forest trained on legitimate transactions only.

    It exposes ``predict_proba`` so it can be scored with exactly the same metrics as the
    supervised models. The "probability" is the forest's anomaly score in [0, 1]; it ranks
    transactions but is not a calibrated probability. Labels are used only to drop fraud
    rows before fitting.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_samples: int | float | str = "auto",
        max_features: float = 1.0,
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> IsolationForestScorer:
        X_arr = np.asarray(X)
        legit = X_arr[np.asarray(y) == 0]
        max_samples = self.max_samples
        if isinstance(max_samples, int):
            max_samples = min(max_samples, len(legit))
        self.forest_ = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=max_samples,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=-1,
        ).fit(legit)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        # score_samples returns the negated anomaly score s(x) in (0, 1].
        score = np.clip(-self.forest_.score_samples(np.asarray(X)), 0.0, 1.0)
        return np.column_stack([1.0 - score, score])

    def predict(self, X: Any) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int64)


def _estimator(name: str, params: dict[str, Any], seed: int, weighted: bool) -> Any:
    if name == LOGISTIC:
        return LogisticRegression(
            max_iter=2000,
            class_weight="balanced" if weighted else None,
            random_state=seed,
            **params,
        )
    if name == FOREST:
        return RandomForestClassifier(
            class_weight="balanced_subsample" if weighted else None,
            random_state=seed,
            n_jobs=-1,
            **params,
        )
    if name == BOOSTING:
        return HistGradientBoostingClassifier(
            class_weight="balanced" if weighted else None,
            early_stopping=False,
            random_state=seed,
            **params,
        )
    if name == ISOLATION:
        return IsolationForestScorer(random_state=seed, **params)
    raise ValueError(f"unknown model {name!r}; expected one of {MODEL_NAMES}")


def build_pipeline(
    name: str,
    params: dict[str, Any],
    seed: int,
    imbalance: ImbalanceConfig | None = None,
) -> Pipeline:
    """Build an unfitted pipeline. ``params['sampling']`` picks the imbalance strategy."""
    imbalance = imbalance or ImbalanceConfig()
    params = dict(params)
    sampling = params.pop("sampling", "none" if name == ISOLATION else "class_weight")
    if sampling not in SAMPLING_STRATEGIES:
        raise ValueError(f"unknown sampling strategy {sampling!r}")
    if name == ISOLATION and sampling != "none":
        raise ValueError("the isolation forest is unsupervised; sampling must be 'none'")
    steps: list[tuple[str, Any]] = [("features", FeatureBuilder()), ("scale", StandardScaler())]
    if sampling == "undersample":
        steps.append(
            (
                "resample",
                RandomUnderSampler(
                    sampling_strategy=imbalance.undersample_ratio, random_state=seed
                ),
            )
        )
    elif sampling == "smote":
        steps.append(
            ("resample", SMOTE(sampling_strategy=imbalance.smote_ratio, random_state=seed))
        )
    steps.append(("model", _estimator(name, params, seed, weighted=sampling == "class_weight")))
    return Pipeline(steps)


def fraud_scores(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Probability (or anomaly score) of fraud for each row."""
    return np.asarray(model.predict_proba(X)[:, 1], dtype=np.float64)


@dataclass
class Trial:
    params: dict[str, Any]
    validation_pr_auc: float
    seconds: float


@dataclass
class SearchResult:
    name: str
    best_params: dict[str, Any]
    best_score: float
    model: Pipeline
    trials: list[Trial] = field(default_factory=list)


def search_on_validation(
    name: str,
    spec: ModelSearch,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    seed: int,
    imbalance: ImbalanceConfig | None = None,
) -> SearchResult:
    """FR-8: randomised search with a fixed seed. Each candidate is trained on the training
    slice and scored by PR-AUC on the validation slice; the test slice is not involved."""
    n_iter = min(spec.n_iter, _grid_size(spec.space))
    candidates = list(ParameterSampler(spec.space, n_iter=n_iter, random_state=seed))
    best: SearchResult | None = None
    trials: list[Trial] = []
    for params in candidates:
        start = time.perf_counter()
        pipe = build_pipeline(name, params, seed, imbalance).fit(X_train, y_train)
        score = float(average_precision_score(y_val, fraud_scores(pipe, X_val)))
        trials.append(Trial(dict(params), score, time.perf_counter() - start))
        if best is None or score > best.best_score:
            best = SearchResult(name, dict(params), score, pipe)
    assert best is not None
    best.trials = trials
    return best


def _grid_size(space: dict[str, list[Any]]) -> int:
    size = 1
    for values in space.values():
        size *= len(values)
    return size


def calibrate(model: Pipeline, X_val: pd.DataFrame, y_val: np.ndarray, method: str) -> Any:
    """FR-9: calibrate a fitted model on validation data without refitting it."""
    return CalibratedClassifierCV(FrozenEstimator(model), method=method).fit(X_val, y_val)
