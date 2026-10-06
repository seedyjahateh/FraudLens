"""Train every model and write versioned artifacts (FR-10).

    python -m fraudlens.train [--config config.yaml]

Only the training and validation slices are read here. Model choice, imbalance strategy,
calibration and the decision threshold are all settled on validation data; the test slice
is left untouched for ``fraudlens.evaluate``.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sklearn.metrics import average_precision_score

from fraudlens.artifacts import (
    SELECTED,
    FraudModel,
    code_version,
    run_version,
    save_model,
    write_json,
)
from fraudlens.config import Config, load_config
from fraudlens.costs import choose_threshold
from fraudlens.data import FEATURE_COLUMNS, file_sha256, load_transactions, time_split, xy
from fraudlens.explain import reference_values
from fraudlens.models import (
    MODEL_NAMES,
    SearchResult,
    build_pipeline,
    calibrate,
    fraud_scores,
    search_on_validation,
)

log = logging.getLogger("fraudlens.train")


def _imbalance_comparison(
    cfg: Config, results: dict[str, SearchResult], X_tr: Any, y_tr: Any, X_va: Any, y_va: Any
) -> list[dict[str, Any]]:
    """EV-5: refit each supervised model's best parameters under every imbalance strategy."""
    rows: list[dict[str, Any]] = []
    for name in cfg.imbalance.models:
        if name not in results:
            continue
        best = results[name]
        base = {k: v for k, v in best.best_params.items() if k != "sampling"}
        for strategy in cfg.imbalance.strategies:
            if best.best_params.get("sampling") == strategy:
                score = best.best_score
                seconds = next(t.seconds for t in best.trials if t.params == best.best_params)
            else:
                start = time.perf_counter()
                pipe = build_pipeline(
                    name, {**base, "sampling": strategy}, cfg.seed, cfg.imbalance
                ).fit(X_tr, y_tr)
                score = float(average_precision_score(y_va, fraud_scores(pipe, X_va)))
                seconds = time.perf_counter() - start
            rows.append(
                {
                    "model": name,
                    "strategy": strategy,
                    "validation_pr_auc": score,
                    "seconds": seconds,
                }
            )
            log.info("imbalance %s/%s: val PR-AUC %.4f", name, strategy, score)
    return rows


def train(cfg: Config, config_text: str) -> str:
    """Run the full training stage and return the artifact version."""
    started = time.perf_counter()
    df = load_transactions(cfg.data_path)
    data_sha = file_sha256(cfg.data_path)
    split = time_split(df, cfg.split)
    split.to_json(cfg.split_index_path, data_sha)
    X_tr, y_tr = xy(df, split.train)
    X_va, y_va = xy(df, split.validation)
    amounts_va = X_va["Amount"].to_numpy()
    version = run_version(data_sha, config_text)
    trained_at = datetime.now(UTC).isoformat(timespec="seconds")
    common = {
        "version": version,
        "trained_at": trained_at,
        "data_sha256": data_sha,
        "code_version": code_version(),
        "seed": cfg.seed,
        "features": FEATURE_COLUMNS,
    }
    log.info(
        "data %s: train %d rows (%d fraud), validation %d rows (%d fraud)",
        data_sha[:8],
        len(y_tr),
        int(y_tr.sum()),
        len(y_va),
        int(y_va.sum()),
    )

    results: dict[str, SearchResult] = {}
    model_summaries: dict[str, Any] = {}
    for name in MODEL_NAMES:
        if name not in cfg.search:
            continue
        t0 = time.perf_counter()
        result = search_on_validation(
            name, cfg.search[name], X_tr, y_tr, X_va, y_va, cfg.seed, cfg.imbalance
        )
        results[name] = result
        val_scores = fraud_scores(result.model, X_va)
        decision = choose_threshold(y_va, val_scores, amounts_va, cfg.costs)
        summary = {
            "model": name,
            "params": result.best_params,
            "validation_pr_auc": result.best_score,
            "threshold": decision.threshold,
            "validation_cost": decision.to_dict(),
            "trials": [asdict(t) for t in result.trials],
            "search_seconds": time.perf_counter() - t0,
        }
        model_summaries[name] = summary
        save_model(cfg.artifacts_dir, name, version, result.model, {**common, **summary})
        log.info(
            "%s: best val PR-AUC %.4f with %s (%.1fs)",
            name,
            result.best_score,
            result.best_params,
            summary["search_seconds"],
        )

    imbalance = _imbalance_comparison(cfg, results, X_tr, y_tr, X_va, y_va)

    # Model selection on validation PR-AUC. The isolation forest competes too; it is
    # only chosen if it genuinely ranks fraud best.
    chosen = max(results.values(), key=lambda r: r.best_score)
    calibrated = calibrate(chosen.model, X_va, y_va, cfg.calibration_method)
    cal_scores = fraud_scores(calibrated, X_va)
    decision = choose_threshold(y_va, cal_scores, amounts_va, cfg.costs)
    selected_meta = {
        **common,
        "model": chosen.name,
        "params": chosen.best_params,
        "calibration": cfg.calibration_method,
        "threshold": decision.threshold,
        "validation_pr_auc": float(average_precision_score(y_va, cal_scores)),
        "validation_cost": decision.to_dict(),
        "selection_rule": "highest validation PR-AUC among all candidates",
    }
    served = FraudModel(
        model_name=chosen.name,
        version=version,
        estimator=calibrated,
        threshold=decision.threshold,
        reference=reference_values(X_tr),
        feature_columns=FEATURE_COLUMNS,
        metadata=selected_meta,
    )
    save_model(cfg.artifacts_dir, SELECTED, version, served, selected_meta)

    write_json(
        cfg.artifacts_dir / "runs" / version / "summary.json",
        {
            **common,
            "split": {
                "train_rows": len(y_tr),
                "train_frauds": int(y_tr.sum()),
                "validation_rows": len(y_va),
                "validation_frauds": int(y_va.sum()),
                "test_rows": len(split.test),
                "split_index_path": cfg.split_index_path,
            },
            "models": model_summaries,
            "imbalance": imbalance,
            "selected": selected_meta,
            "train_seconds": time.perf_counter() - started,
        },
    )
    (cfg.artifacts_dir / "LATEST").write_text(version + "\n", encoding="utf-8")
    log.info(
        "selected %s (calibrated, %s), threshold %.4f, version %s, %.1fs",
        chosen.name,
        cfg.calibration_method,
        decision.threshold,
        version,
        time.perf_counter() - started,
    )
    return version


def main(argv: list[str] | None = None) -> str:
    parser = argparse.ArgumentParser(description="Train all FraudLens models.")
    parser.add_argument("--config", default="config.yaml", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    return train(load_config(args.config), args.config.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
