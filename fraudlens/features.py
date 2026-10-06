"""Feature engineering shared by training and serving (FR-3).

The fitted model pipeline starts with :class:`FeatureBuilder`, so the service applies
exactly the code that training used; nothing is reimplemented on the serving side.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from fraudlens.data import FEATURE_COLUMNS, PCA_COLUMNS

SECONDS_PER_HOUR = 3600
ENGINEERED_COLUMNS: list[str] = [*PCA_COLUMNS, "log_amount", "hour"]


def build_features(X: pd.DataFrame) -> pd.DataFrame:
    """Raw transaction columns -> model features.

    - ``log_amount`` = log1p(Amount); scaling happens in the fitted pipeline.
    - ``hour`` = hour of day derived from ``Time`` (seconds since the first transaction).
    - ``V1``-``V28`` pass through unchanged.
    """
    missing = [c for c in FEATURE_COLUMNS if c not in X.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    out = X[PCA_COLUMNS].astype(np.float64).copy()
    out["log_amount"] = np.log1p(X["Amount"].astype(np.float64).clip(lower=0))
    out["hour"] = (X["Time"].astype(np.float64) // SECONDS_PER_HOUR) % 24
    return out[ENGINEERED_COLUMNS]


def transaction_frame(transaction: dict[str, float]) -> pd.DataFrame:
    """One transaction as a single-row frame in the canonical column order (serving path)."""
    return pd.DataFrame([{c: float(transaction[c]) for c in FEATURE_COLUMNS}])


class FeatureBuilder(TransformerMixin, BaseEstimator):
    """Stateless scikit-learn wrapper around :func:`build_features`."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> FeatureBuilder:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return build_features(X)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(ENGINEERED_COLUMNS, dtype=object)
