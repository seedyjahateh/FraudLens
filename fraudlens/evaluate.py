"""Evaluate every trained model on the held-out test slice and write the report (FR-11).

    python -m fraudlens.evaluate [--config config.yaml]

This is the only module that reads test rows. By the time it runs, the model, the
imbalance strategy, the calibration and the threshold have all been fixed on validation
data, so the test slice is used once, for reporting.
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score

from fraudlens import plots, report
from fraudlens.artifacts import (
    SELECTED,
    latest_version,
    load_model,
    read_json,
    write_json,
)
from fraudlens.config import Config, EvaluationConfig, load_config
from fraudlens.costs import baseline_costs, cost_at_threshold, cost_curve, savings
from fraudlens.data import Split, file_sha256, load_transactions, xy
from fraudlens.explain import global_importance
from fraudlens.models import MODEL_NAMES, SUPERVISED, fraud_scores

log = logging.getLogger("fraudlens.evaluate")

PRIMARY = ["pr_auc", "recall_at_precision", "precision_at_recall"]


def pr_metrics(y: np.ndarray, s: np.ndarray, p_target: float, r_target: float) -> dict[str, float]:
    """PR-AUC (average precision), best recall at precision >= p_target and best precision
    at recall >= r_target, from a single precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y, s)
    ap = float(-np.sum(np.diff(recall) * precision[:-1]))
    r_at_p = recall[precision >= p_target]
    p_at_r = precision[recall >= r_target]
    return {
        "pr_auc": ap,
        "recall_at_precision": float(r_at_p.max()) if r_at_p.size else 0.0,
        "precision_at_recall": float(p_at_r.max()) if p_at_r.size else 0.0,
    }


def score_metrics(
    y: np.ndarray, s: np.ndarray, amounts: np.ndarray, threshold: float, cfg: Config
) -> dict[str, float]:
    """Every per-model test metric that gets a bootstrap interval."""
    ev = cfg.evaluation
    out = pr_metrics(y, s, ev.recall_at_precision, ev.precision_at_recall)
    out["roc_auc"] = float(roc_auc_score(y, s))
    out["brier"] = float(np.mean((s - y) ** 2))
    outcome = cost_at_threshold(y, s, amounts, threshold, cfg.costs)
    n = len(y)
    flagged = outcome.tp + outcome.fp
    out["accuracy"] = (outcome.tp + outcome.tn) / n
    out["precision_at_threshold"] = outcome.tp / flagged if flagged else 0.0
    out["recall_at_threshold"] = outcome.tp / max(outcome.tp + outcome.fn, 1)
    out["total_cost"] = outcome.total_cost
    base = baseline_costs(y, amounts, cfg.costs)
    out["savings_vs_flag_nothing_pct"] = savings(outcome.total_cost, base["flag_nothing"])[1]
    out["savings_vs_flag_everything_pct"] = savings(outcome.total_cost, base["flag_everything"])[1]
    return out


@dataclass
class Bootstrap:
    """Paired bootstrap: every model is scored on the same resampled index sets."""

    point: dict[str, dict[str, float]]
    samples: dict[str, dict[str, np.ndarray]]

    def interval(self, model: str, metric: str, confidence: float) -> tuple[float, float]:
        values = self.samples[model][metric]
        alpha = (1.0 - confidence) / 2.0
        return float(np.quantile(values, alpha)), float(np.quantile(values, 1.0 - alpha))


def bootstrap(
    y: np.ndarray,
    amounts: np.ndarray,
    scored: dict[str, tuple[np.ndarray, float]],
    metric_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, float], dict[str, float]],
    n_samples: int,
    seed: int,
) -> Bootstrap:
    """EV-4: percentile bootstrap over test rows. Resamples without any fraud are redrawn."""
    rng = np.random.default_rng(seed)
    point = {m: metric_fn(y, s, amounts, t) for m, (s, t) in scored.items()}
    collected: dict[str, dict[str, list[float]]] = {m: {k: [] for k in point[m]} for m in scored}
    n = len(y)
    done = 0
    while done < n_samples:
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        for m, (s, t) in scored.items():
            for k, v in metric_fn(yb, s[idx], amounts[idx], t).items():
                collected[m][k].append(v)
        done += 1
    samples = {m: {k: np.asarray(v) for k, v in d.items()} for m, d in collected.items()}
    return Bootstrap(point, samples)


def metric_label(metric: str, ev: EvaluationConfig) -> str:
    if metric == "recall_at_precision":
        return f"recall_at_{round(ev.recall_at_precision * 100)}pct_precision"
    if metric == "precision_at_recall":
        return f"precision_at_{round(ev.precision_at_recall * 100)}pct_recall"
    return metric


def evaluate(cfg: Config, version: str | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    version = version or latest_version(cfg.artifacts_dir)
    summary = read_json(cfg.artifacts_dir / "runs" / version / "summary.json")
    df = load_transactions(cfg.data_path)
    data_sha = file_sha256(cfg.data_path)
    if data_sha != summary["data_sha256"]:
        raise ValueError("the data file changed since training; re-run training first")
    split = Split.from_json(cfg.split_index_path, data_sha)
    X_va, y_va = xy(df, split.validation)
    X_te, y_te = xy(df, split.test)
    amounts_va = X_va["Amount"].to_numpy()
    amounts_te = X_te["Amount"].to_numpy()

    selected_dir = cfg.artifacts_dir / SELECTED / version
    if (selected_dir / "evaluation.json").exists():
        log.warning(
            "test slice already evaluated for version %s; recomputing the same numbers. "
            "Do not change the model in response to them.",
            version,
        )

    served, served_meta = load_model(cfg.artifacts_dir, SELECTED, version)
    chosen_name = served_meta["model"]
    served_label = f"{chosen_name} (calibrated)"

    # Score every model on validation (for curves) and test (for headline numbers).
    val_scores: dict[str, np.ndarray] = {}
    test_scored: dict[str, tuple[np.ndarray, float]] = {}
    candidates: dict[str, Any] = {}
    for name in MODEL_NAMES:
        if name not in summary["models"]:
            continue
        model, meta = load_model(cfg.artifacts_dir, name, version)
        candidates[name] = model
        val_scores[name] = fraud_scores(model, X_va)
        test_scored[name] = (fraud_scores(model, X_te), float(meta["threshold"]))
    val_scores[served_label] = served.predict_proba(X_va)
    test_scored[served_label] = (served.predict_proba(X_te), float(served.threshold))

    def metric_fn(y: np.ndarray, s: np.ndarray, a: np.ndarray, t: float) -> dict[str, float]:
        return score_metrics(y, s, a, t, cfg)

    boot = bootstrap(
        y_te, amounts_te, test_scored, metric_fn, cfg.evaluation.bootstrap_samples, cfg.seed
    )

    rows: list[dict[str, Any]] = []
    conf = cfg.evaluation.confidence
    for name, (scores, threshold) in test_scored.items():
        role = "served" if name == served_label else "candidate"
        val_pr_auc = (
            summary["models"][name]["validation_pr_auc"]
            if name in summary["models"]
            else served_meta["validation_pr_auc"]
        )
        rows.append(_row(name, role, "validation", "pr_auc", val_pr_auc))
        rows.append(_row(name, role, "validation", "threshold", threshold))
        for metric, value in boot.point[name].items():
            low, high = boot.interval(name, metric, conf)
            rows.append(
                _row(name, role, "test", metric_label(metric, cfg.evaluation), value, low, high)
            )
        outcome = cost_at_threshold(y_te, scores, amounts_te, threshold, cfg.costs)
        for k in ("tp", "fp", "fn", "tn"):
            rows.append(_row(name, role, "test", k, getattr(outcome, k)))

    base_test = baseline_costs(y_te, amounts_te, cfg.costs)
    always_legit_acc = float(np.mean(y_te == 0))
    rows.append(_row("always_legit", "baseline", "test", "accuracy", always_legit_acc))
    rows.append(_row("always_legit", "baseline", "test", "recall_at_threshold", 0.0))
    rows.append(_row("always_legit", "baseline", "test", "total_cost", base_test["flag_nothing"]))
    rows.append(
        _row("flag_everything", "baseline", "test", "total_cost", base_test["flag_everything"])
    )
    metrics = pd.DataFrame(rows)

    # Paired differences in PR-AUC: the served model against every other candidate. Its own
    # uncalibrated version is skipped: monotonic calibration leaves PR-AUC unchanged.
    comparisons = []
    for other in test_scored:
        if other in (served_label, chosen_name):
            continue
        diff = boot.samples[served_label]["pr_auc"] - boot.samples[other]["pr_auc"]
        alpha = (1.0 - conf) / 2.0
        low, high = float(np.quantile(diff, alpha)), float(np.quantile(diff, 1.0 - alpha))
        comparisons.append(
            {
                "model_a": served_label,
                "model_b": other,
                "metric": "pr_auc",
                "difference": boot.point[served_label]["pr_auc"] - boot.point[other]["pr_auc"],
                "ci_low": low,
                "ci_high": high,
                "significant": bool(low > 0 or high < 0),
            }
        )
    comparisons_df = pd.DataFrame(comparisons)

    served_scores, served_threshold = test_scored[served_label]
    served_outcome = cost_at_threshold(y_te, served_scores, amounts_te, served_threshold, cfg.costs)
    cost_summary = {
        "threshold": served_threshold,
        "model_cost": served_outcome.total_cost,
        "flag_nothing_cost": base_test["flag_nothing"],
        "flag_everything_cost": base_test["flag_everything"],
        "saved_vs_flag_nothing": savings(served_outcome.total_cost, base_test["flag_nothing"]),
        "saved_vs_flag_everything": savings(
            served_outcome.total_cost, base_test["flag_everything"]
        ),
        "outcome": served_outcome.to_dict(),
        "validation_outcome": served_meta["validation_cost"],
        "test_frauds": int(y_te.sum()),
        "test_rows": len(y_te),
        "test_fraud_amount": float(amounts_te[y_te == 1].sum()),
    }

    importance = global_importance(
        candidates[chosen_name], X_va, y_va, cfg.permutation_repeats, cfg.seed
    )

    reports = cfg.reports_dir
    reports.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(reports / "metrics.csv", index=False, float_format="%.6g", lineterminator="\n")
    comparisons_df.to_csv(
        reports / "comparisons.csv", index=False, float_format="%.6g", lineterminator="\n"
    )
    pd.DataFrame(summary["imbalance"]).drop(columns=["seconds"], errors="ignore").to_csv(
        reports / "imbalance.csv", index=False, float_format="%.6g", lineterminator="\n"
    )
    importance.to_csv(
        reports / "feature_importance.csv", index=False, float_format="%.6g", lineterminator="\n"
    )
    write_json(
        reports / "run.json",
        {
            "version": version,
            "data_sha256": data_sha,
            "bootstrap_samples": cfg.evaluation.bootstrap_samples,
            "confidence": conf,
            "cost_summary": cost_summary,
        },
    )
    plots.pr_curves(y_te, {k: v[0] for k, v in test_scored.items()}, reports / "pr_curves.png")
    plots.calibration(
        y_te,
        # The isolation forest's score is an anomaly score, not a probability.
        {k: v[0] for k, v in test_scored.items() if k in (*SUPERVISED, served_label)},
        reports / "calibration.png",
    )
    val_curve = cost_curve(y_va, val_scores[served_label], amounts_va, cfg.costs)
    test_curve = cost_curve(y_te, served_scores, amounts_te, cfg.costs)
    plots.cost_curves(val_curve, test_curve, served_threshold, reports / "cost_curve.png")
    plots.importance(importance, reports / "feature_importance.png")

    evaluation = {
        "version": version,
        "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": chosen_name,
        "threshold": served_threshold,
        "test_pr_auc": boot.point[served_label]["pr_auc"],
        "test_pr_auc_ci": list(boot.interval(served_label, "pr_auc", conf)),
        "test_rows": len(y_te),
    }
    write_json(selected_dir / "evaluation.json", evaluation)

    context = report.ReportContext(
        cfg=cfg,
        summary=summary,
        metrics=metrics,
        comparisons=comparisons_df,
        importance=importance,
        costs=cost_summary,
        served_label=served_label,
        evaluate_seconds=time.perf_counter() - started,
    )
    report.write_report(context, reports / "report.md")
    report.update_readme(cfg.readme_path, reports)
    log.info("wrote reports to %s in %.1fs", reports, time.perf_counter() - started)
    return evaluation


def _row(
    model: str,
    role: str,
    split: str,
    metric: str,
    value: float,
    low: float | None = None,
    high: float | None = None,
) -> dict[str, Any]:
    return {
        "model": model,
        "role": role,
        "split": split,
        "metric": metric,
        "value": value,
        "ci_low": low,
        "ci_high": high,
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Evaluate FraudLens models on the test slice.")
    parser.add_argument("--config", default="config.yaml", type=Path)
    parser.add_argument("--version", default=None, help="artifact version (default: latest)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    return evaluate(load_config(args.config), args.version)


if __name__ == "__main__":
    main()
