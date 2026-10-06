"""Cost model and threshold search (EV-7).

Per transaction, with a decision "flag if score >= threshold":

- a missed fraud (false negative) costs its amount (or a fixed figure, per config);
- a false alarm (false positive) costs one review;
- a caught fraud (true positive) costs one review too (configurable, may be 0);
- a correctly allowed legitimate transaction costs nothing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from fraudlens.config import CostConfig


@dataclass(frozen=True)
class CostOutcome:
    threshold: float
    total_cost: float
    tp: int
    fp: int
    fn: int
    tn: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def missed_costs(y: np.ndarray, amounts: np.ndarray, costs: CostConfig) -> np.ndarray:
    """Cost of letting each transaction through (non-zero only for fraud)."""
    per = (
        np.asarray(amounts, dtype=np.float64)
        if costs.missed_fraud == "amount"
        else np.full(len(y), float(costs.missed_fraud))
    )
    return np.where(np.asarray(y) == 1, per, 0.0)


def cost_at_threshold(
    y: np.ndarray, scores: np.ndarray, amounts: np.ndarray, threshold: float, costs: CostConfig
) -> CostOutcome:
    y = np.asarray(y)
    flag = np.asarray(scores) >= threshold
    fraud = y == 1
    tp = int(np.sum(flag & fraud))
    fp = int(np.sum(flag & ~fraud))
    fn = int(np.sum(~flag & fraud))
    tn = int(np.sum(~flag & ~fraud))
    total = (
        float(np.sum(missed_costs(y, amounts, costs)[~flag]))
        + fp * costs.false_alarm
        + tp * costs.caught_fraud
    )
    return CostOutcome(float(threshold), total, tp, fp, fn, tn)


def cost_curve(
    y: np.ndarray, scores: np.ndarray, amounts: np.ndarray, costs: CostConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Total cost for every distinct threshold, from "flag nothing" to "flag everything".

    Returns ``(thresholds, total_costs)`` with thresholds in decreasing order. The first
    threshold lies just above the highest score (nothing flagged).
    """
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind="mergesort")
    s = scores[order]
    ys = y[order]
    missed = missed_costs(ys, np.asarray(amounts)[order], costs)
    flag_cost = np.where(ys == 1, costs.caught_fraud, costs.false_alarm)
    # Flagging the top k rows: pay review costs for them, save their missed-fraud cost.
    delta = np.cumsum(flag_cost - missed)
    base = float(missed.sum())
    # Valid cut points are the ends of runs of equal scores (ties flag together).
    ends = np.flatnonzero(np.append(s[1:] != s[:-1], True))
    thresholds = np.concatenate([[np.nextafter(s[0], np.inf)], s[ends]])
    totals = np.concatenate([[base], base + delta[ends]])
    return thresholds, totals


def choose_threshold(
    y: np.ndarray, scores: np.ndarray, amounts: np.ndarray, costs: CostConfig
) -> CostOutcome:
    """The threshold that minimises total cost. Ties go to the higher threshold."""
    thresholds, totals = cost_curve(y, scores, amounts, costs)
    best = int(np.argmin(totals))
    return cost_at_threshold(y, scores, amounts, float(thresholds[best]), costs)


def baseline_costs(y: np.ndarray, amounts: np.ndarray, costs: CostConfig) -> dict[str, float]:
    """Cost of the two trivial policies the model must beat."""
    y = np.asarray(y)
    n_fraud = int(np.sum(y == 1))
    return {
        "flag_nothing": float(missed_costs(y, amounts, costs).sum()),
        "flag_everything": (len(y) - n_fraud) * costs.false_alarm + n_fraud * costs.caught_fraud,
    }


def savings(model_cost: float, baseline_cost: float) -> tuple[float, float]:
    """Money saved against a baseline, and as a percentage of that baseline."""
    saved = baseline_cost - model_cost
    pct = 100.0 * saved / baseline_cost if baseline_cost > 0 else 0.0
    return saved, pct
