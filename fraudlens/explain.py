"""Explainability (FR-13, FR-14).

Global: permutation importance on the validation slice, scored by PR-AUC.

Per transaction: reference substitution. For each input feature, replace the transaction's
value with the training median and re-score; the contribution is how much the fraud
probability drops (positive = the feature pushed towards fraud). All substitutions are
scored in one batch, so an explanation costs a single extra ``predict_proba`` call.

The inputs ``V1``-``V28`` are anonymised PCA components, so explanations name components
rather than human-readable concepts. ``Time`` matters to the model only through the
hour of day and ``Amount`` through log1p(Amount).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from fraudlens.data import FEATURE_COLUMNS
from fraudlens.models import fraud_scores


@dataclass(frozen=True)
class Contribution:
    feature: str
    contribution: float


def reference_values(X_train: pd.DataFrame) -> dict[str, float]:
    """Training medians: the "typical transaction" each feature is compared against."""
    return {c: float(X_train[c].median()) for c in FEATURE_COLUMNS}


def explain_row(
    model: Any, row: pd.DataFrame, reference: dict[str, float], top_k: int = 5
) -> tuple[float, list[Contribution]]:
    """Fraud probability for a single-row frame and its ``top_k`` contributing features."""
    if len(row) != 1:
        raise ValueError("explain_row expects exactly one transaction")
    base = row[FEATURE_COLUMNS].reset_index(drop=True)
    batch = pd.concat([base] * (len(FEATURE_COLUMNS) + 1), ignore_index=True)
    for i, col in enumerate(FEATURE_COLUMNS, start=1):
        batch.loc[i, col] = reference[col]
    scores = fraud_scores(model, batch)
    prob = float(scores[0])
    deltas = prob - scores[1:]
    order = np.argsort(-np.abs(deltas), kind="mergesort")[:top_k]
    return prob, [Contribution(FEATURE_COLUMNS[i], float(deltas[i])) for i in order]


def global_importance(
    model: Any,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    n_repeats: int,
    seed: int,
) -> pd.DataFrame:
    """Permutation importance (drop in validation PR-AUC when a column is shuffled)."""
    result = permutation_importance(
        model,
        X_val[FEATURE_COLUMNS],
        y_val,
        scoring="average_precision",
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=1,
    )
    frame = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return frame.sort_values("importance_mean", ascending=False, ignore_index=True)
