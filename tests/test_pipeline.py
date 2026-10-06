"""End to end on the synthetic fixture: train -> evaluate -> report (FR-10..FR-12, NFR-4)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraudlens import evaluate as evaluate_mod
from fraudlens import report
from fraudlens import train as train_mod
from fraudlens.artifacts import SELECTED, latest_version, load_selected, read_json
from fraudlens.config import load_config
from fraudlens.evaluate import pr_metrics
from tests.conftest import write_small_config

REPORT_FILES = [
    "metrics.csv",
    "comparisons.csv",
    "imbalance.csv",
    "feature_importance.csv",
    "pr_curves.png",
    "calibration.png",
    "cost_curve.png",
    "feature_importance.png",
    "report.md",
    "run.json",
]


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("run")
    cfg_path = write_small_config(tmp)
    train_mod.main(["--config", str(cfg_path)])
    evaluate_mod.main(["--config", str(cfg_path)])
    return cfg_path


def test_artifacts_are_versioned_with_provenance(run: Path) -> None:
    cfg = load_config(run)
    version = latest_version(cfg.artifacts_dir)
    for name in [
        "logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
        "isolation_forest",
        SELECTED,
    ]:
        meta = read_json(cfg.artifacts_dir / name / version / "metadata.json")
        for key in ("version", "data_sha256", "code_version", "seed", "features", "threshold"):
            assert key in meta, (name, key)
        assert (cfg.artifacts_dir / name / version / "model.joblib").exists()
    served = load_selected(cfg.artifacts_dir / SELECTED / version)
    assert served.metadata["evaluation"]["version"] == version
    assert 0.0 <= served.metadata["evaluation"]["test_pr_auc"] <= 1.0


def test_reports_written(run: Path) -> None:
    reports = load_config(run).reports_dir
    for name in REPORT_FILES:
        assert (reports / name).exists(), name
    text = (reports / "report.md").read_text(encoding="utf-8")
    for heading in ("Split", "Imbalance", "Cost-based threshold", "Limitations"):
        assert heading in text


def test_metrics_csv_has_cis_for_every_model(run: Path) -> None:
    metrics = pd.read_csv(load_config(run).reports_dir / "metrics.csv")
    served = metrics[metrics["role"] == "served"]["model"].unique()
    assert len(served) == 1
    candidates = metrics[metrics["role"] == "candidate"]["model"].unique()
    assert len(candidates) == 4
    for model in [*served, *candidates]:
        rows = metrics[(metrics["model"] == model) & (metrics["split"] == "test")]
        for metric in ("pr_auc", "recall_at_90pct_precision", "precision_at_80pct_recall"):
            r = rows[rows["metric"] == metric].iloc[0]
            assert r["ci_low"] <= r["value"] <= r["ci_high"], (model, metric)
    acc = metrics[(metrics["model"] == "always_legit") & (metrics["metric"] == "accuracy")]
    assert acc["value"].iloc[0] > 0.9


def test_readme_generated_from_metrics(run: Path) -> None:
    cfg = load_config(run)
    readme = cfg.readme_path.read_text(encoding="utf-8")
    metrics = pd.read_csv(cfg.reports_dir / "metrics.csv")
    served = metrics[metrics["role"] == "served"]["model"].iloc[0]
    pr_auc = metrics[
        (metrics["model"] == served)
        & (metrics["metric"] == "pr_auc")
        & (metrics["split"] == "test")
    ]["value"].iloc[0]
    assert f"{pr_auc:.3f}" in readme
    assert "Do not edit by hand" in readme
    # Regenerating is idempotent.
    report.main(["--config", str(run)])
    assert cfg.readme_path.read_text(encoding="utf-8") == readme


def test_rerun_is_deterministic(run: Path, tmp_path: Path) -> None:
    """NFR-4: same seed and data, same metrics."""
    first = pd.read_csv(load_config(run).reports_dir / "metrics.csv")
    again_cfg = write_small_config(tmp_path)
    train_mod.main(["--config", str(again_cfg)])
    evaluate_mod.main(["--config", str(again_cfg)])
    second = pd.read_csv(load_config(again_cfg).reports_dir / "metrics.csv")
    pd.testing.assert_frame_equal(first, second, rtol=1e-9)


def test_reevaluation_warns(run: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("WARNING")
    evaluate_mod.evaluate(load_config(run))
    assert "already evaluated" in caplog.text


def test_changed_data_is_refused(run: Path, tmp_path: Path) -> None:
    cfg = load_config(run)
    other = tmp_path / "other.csv"
    df = pd.read_csv(cfg.data_path)
    df.loc[0, "Amount"] += 1
    df.to_csv(other, index=False)
    from dataclasses import replace

    with pytest.raises(ValueError, match="changed since training"):
        evaluate_mod.evaluate(replace(cfg, data_path=other))


def test_pr_metrics_known_values() -> None:
    y = np.array([1, 0, 1, 0])
    s = np.array([0.9, 0.8, 0.7, 0.1])
    m = pr_metrics(y, s, p_target=0.9, r_target=0.8)
    assert m["pr_auc"] == pytest.approx((1.0 + 2 / 3) / 2)
    assert m["recall_at_precision"] == pytest.approx(0.5)
    assert m["precision_at_recall"] == pytest.approx(2 / 3)


def test_readme_markers_required(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="markers"):
        report.replace_section("no markers here", report.RESULTS, "x")


def test_latency_section(tmp_path: Path) -> None:
    assert "No load test" in report.latency_text(tmp_path)
    (tmp_path / "load_test.json").write_text(
        json.dumps(
            {
                "requests": 1000,
                "target": "http://x",
                "environment": "test",
                "p50_ms": 3.0,
                "p95_ms": 7.5,
                "p99_ms": 9.0,
                "max_ms": 20.0,
                "errors": 0,
                "budget_ms": 50,
                "measured_at": "now",
            }
        ),
        encoding="utf-8",
    )
    assert "7.5 ms" in report.latency_text(tmp_path)
    assert "met" in report.latency_text(tmp_path)


def test_main_module_runs_everything(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from fraudlens import __main__ as entry

    cfg_path = write_small_config(tmp_path)
    monkeypatch.setattr(sys, "argv", ["fraudlens", "--config", str(cfg_path)])
    entry.main()
    assert (load_config(cfg_path).reports_dir / "report.md").exists()
