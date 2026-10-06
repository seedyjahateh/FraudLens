"""The data bundle behind the web dashboard (``dashboard/``).

``evaluate`` writes it to ``reports/dashboard.json`` and next to the served model in
``artifacts/selected/<version>/dashboard.json``, so the API always serves the bundle that
belongs to the model it loaded. The bundle holds only derived numbers (metrics, curves,
confusion counts); it never contains dataset rows.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve

from fraudlens.config import Config
from fraudlens.costs import Sweep, cost_sweep
from fraudlens.models import SUPERVISED
from fraudlens.report import better_unselected

SCHEMA_VERSION = 1
SWEEP_HEAD = 400  # keep every cut point while few transactions are flagged
SWEEP_TAIL = 200  # then log-spaced points out to "flag everything"
HIST_BINS = 20


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """DataFrame rows as JSON-safe dicts (NaN -> None, numpy -> Python)."""
    clean = frame.astype(object).where(pd.notna(frame), None)
    return [{str(k): _py(v) for k, v in row.items()} for row in clean.to_dict("records")]


def _py(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _round(values: np.ndarray, digits: int = 6) -> list[float]:
    return [float(v) for v in np.round(np.asarray(values, dtype=np.float64), digits)]


def pr_curve(y: np.ndarray, scores: np.ndarray) -> dict[str, list[float]]:
    """Precision-recall curve reduced to its corners: recall only changes when a fraud is
    crossed, so keeping the first and last point of each recall run is exact."""
    precision, recall, _ = precision_recall_curve(y, scores)
    change = np.flatnonzero(np.diff(recall) != 0)
    keep = np.unique(np.concatenate([[0, len(recall) - 1], change, change + 1]))
    return {"recall": _round(recall[keep], 5), "precision": _round(precision[keep], 5)}


def _sweep_indices(sweep: Sweep, always: list[int]) -> np.ndarray:
    n = len(sweep.thresholds)
    head = np.arange(min(SWEEP_HEAD, n))
    tail = (
        np.unique(np.geomspace(SWEEP_HEAD, n - 1, SWEEP_TAIL).astype(np.int64))
        if n > SWEEP_HEAD
        else np.array([], dtype=np.int64)
    )
    return np.unique(np.concatenate([head, tail, [n - 1], always])).astype(np.int64)


def sweep_payload(
    y: np.ndarray, scores: np.ndarray, amounts: np.ndarray, cfg: Config, chosen: float
) -> dict[str, Any]:
    """Confusion counts and missed-fraud cost at (a sample of) every threshold.

    ``chosen_index`` points at the sample whose flags equal "score >= chosen threshold";
    ``min_cost_index`` at the cheapest sample under the configured costs.
    """
    sweep = cost_sweep(y, scores, amounts, cfg.costs)
    # Thresholds decrease; the last one still >= chosen flags exactly the rows >= chosen.
    chosen_full = int(np.flatnonzero(sweep.thresholds >= chosen)[-1])
    min_full = int(np.argmin(sweep.totals(cfg.costs)))
    idx = _sweep_indices(sweep, [chosen_full, min_full])
    return {
        # Full precision: rounding could move a threshold across a score and change the
        # flag counts. The first value may exceed 1 ("flag nothing").
        "threshold": [float(t) for t in sweep.thresholds[idx]],
        "tp": sweep.tp[idx].tolist(),
        "fp": sweep.fp[idx].tolist(),
        "fn": sweep.fn[idx].tolist(),
        "tn": sweep.tn[idx].tolist(),
        "missed_cost": _round(sweep.missed_cost[idx], 2),
        "chosen_index": int(np.searchsorted(idx, chosen_full)),
        "min_cost_index": int(np.searchsorted(idx, min_full)),
        "chosen_threshold": float(chosen),
        "rows": len(y),
        "frauds": int(np.sum(y == 1)),
        "fraud_amount": float(np.sum(np.asarray(amounts)[np.asarray(y) == 1])),
    }


def calibration_payload(y: np.ndarray, scores: np.ndarray, bins: int) -> dict[str, Any]:
    obs, pred = calibration_curve(y, scores, n_bins=bins, strategy="uniform")
    counts, _ = np.histogram(scores, bins=HIST_BINS, range=(0.0, 1.0))
    return {
        "predicted": _round(pred),
        "observed": _round(obs),
        "histogram": counts.tolist(),
    }


def build_bundle(
    *,
    cfg: Config,
    summary: dict[str, Any],
    evaluation: dict[str, Any],
    metrics: pd.DataFrame,
    comparisons: pd.DataFrame,
    importance: pd.DataFrame,
    cost_summary: dict[str, Any],
    served_label: str,
    y_val: np.ndarray,
    amounts_val: np.ndarray,
    val_scores: dict[str, np.ndarray],
    y_test: np.ndarray,
    amounts_test: np.ndarray,
    test_scores: dict[str, np.ndarray],
) -> dict[str, Any]:
    selected = summary["selected"]
    sp = summary["split"]
    threshold = float(selected["threshold"])
    return {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "version": summary["version"],
            "data_sha256": summary["data_sha256"],
            "code_version": summary["code_version"],
            "seed": summary["seed"],
            "trained_at": summary["trained_at"],
            "evaluated_at": evaluation["evaluated_at"],
            "train_seconds": summary["train_seconds"],
            "bootstrap_samples": cfg.evaluation.bootstrap_samples,
            "confidence": cfg.evaluation.confidence,
            "served_model": selected["model"],
            "served_label": served_label,
            "calibration": selected["calibration"],
            "selection_rule": selected["selection_rule"],
            "threshold": threshold,
        },
        "split": {
            "train": {"rows": sp["train_rows"], "frauds": sp["train_frauds"]},
            "validation": {"rows": sp["validation_rows"], "frauds": sp["validation_frauds"]},
            "test": {"rows": cost_summary["test_rows"], "frauds": cost_summary["test_frauds"]},
        },
        "metrics": _records(metrics),
        "comparisons": _records(comparisons),
        "finding": {"better_unselected": better_unselected(comparisons)},
        "pr_curves": {
            "prevalence": float(np.mean(y_test)),
            "models": {name: pr_curve(y_test, s) for name, s in test_scores.items()},
        },
        "sweeps": {
            "validation": sweep_payload(
                y_val, val_scores[served_label], amounts_val, cfg, threshold
            ),
            "test": sweep_payload(y_test, test_scores[served_label], amounts_test, cfg, threshold),
        },
        "calibration": {
            name: calibration_payload(y_test, s, cfg.evaluation.calibration_bins)
            for name, s in test_scores.items()
            if name in (*SUPERVISED, served_label)
        },
        "costs": {
            "config": {
                "missed_fraud": cfg.costs.missed_fraud,
                "false_alarm": cfg.costs.false_alarm,
                "caught_fraud": cfg.costs.caught_fraud,
            },
            "test": {
                "model_cost": cost_summary["model_cost"],
                "flag_nothing_cost": cost_summary["flag_nothing_cost"],
                "flag_everything_cost": cost_summary["flag_everything_cost"],
            },
            "validation_outcome": cost_summary["validation_outcome"],
        },
        "importance": _records(importance),
        "imbalance": [
            {k: v for k, v in row.items() if k != "seconds"} for row in summary["imbalance"]
        ],
        "search": {
            name: {
                "params": m["params"],
                "validation_pr_auc": m["validation_pr_auc"],
                "threshold": m["threshold"],
                "search_seconds": m["search_seconds"],
                "trials": m["trials"],
            }
            for name, m in summary["models"].items()
        },
    }
