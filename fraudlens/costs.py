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


@dataclass(frozen=True)
class Sweep:
    """Confusion counts at every distinct threshold, thresholds in decreasing order.

    ``missed_cost`` is the cost of the frauds let through (per the ``missed_fraud`` rule),
    so the total for any review costs is ``missed_cost + fp*false_alarm + tp*caught_fraud``.
    """

    thresholds: np.ndarray
    tp: np.ndarray
    fp: np.ndarray
    fn: np.ndarray
    tn: np.ndarray
    missed_cost: np.ndarray

    def totals(self, costs: CostConfig) -> np.ndarray:
        return self.missed_cost + self.fp * costs.false_alarm + self.tp * costs.caught_fraud


def cost_sweep(y: np.ndarray, scores: np.ndarray, amounts: np.ndarray, costs: CostConfig) -> Sweep:
    """Flag the top k rows for every valid k. The first threshold lies just above the
    highest score (nothing flagged); ties in score are flagged together."""
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind="mergesort")
    s = scores[order]
    fraud = (y[order] == 1).astype(np.int64)
    missed = missed_costs(y[order], np.asarray(amounts)[order], costs)
    ends = np.flatnonzero(np.append(s[1:] != s[:-1], True))
    tp = np.concatenate([[0], np.cumsum(fraud)[ends]])
    flagged = np.concatenate([[0], ends + 1])
    fp = flagged - tp
    n_fraud = int(fraud.sum())
    return Sweep(
        thresholds=np.concatenate([[np.nextafter(s[0], np.inf)], s[ends]]),
        tp=tp,
        fp=fp,
        fn=n_fraud - tp,
        tn=(len(s) - n_fraud) - fp,
        missed_cost=float(missed.sum()) - np.concatenate([[0.0], np.cumsum(missed)[ends]]),
    )


def cost_curve(
    y: np.ndarray, scores: np.ndarray, amounts: np.ndarray, costs: CostConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Total cost for every distinct threshold, from "flag nothing" to "flag everything"."""
    sweep = cost_sweep(y, scores, amounts, costs)
    return sweep.thresholds, sweep.totals(costs)


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
