"""The dashboard data bundle written by evaluate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from fraudlens.artifacts import SELECTED, latest_version
from fraudlens.config import load_config
from fraudlens.costs import cost_at_threshold
from fraudlens.dashboard_data import SCHEMA_VERSION, pr_curve, sweep_payload
from fraudlens.evaluate import evaluate
from fraudlens.train import train
from tests.conftest import write_small_config
from tests.fixtures.make_fixture import FIXTURE_PATH

TOP_LEVEL = {
    "schema_version",
    "meta",
    "split",
    "metrics",
    "comparisons",
    "finding",
    "pr_curves",
    "sweeps",
    "calibration",
    "costs",
    "importance",
    "imbalance",
    "search",
}


@pytest.fixture(scope="module")
def bundle_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any]]:
    cfg_path = write_small_config(tmp_path_factory.mktemp("dash"))
    cfg = load_config(cfg_path)
    train(cfg, cfg_path.read_text(encoding="utf-8"))
    evaluate(cfg)
    bundle = json.loads((cfg.reports_dir / "dashboard.json").read_text(encoding="utf-8"))
    return cfg_path, bundle


def test_bundle_shape(bundle_run: tuple[Path, dict[str, Any]]) -> None:
    _, bundle = bundle_run
    assert set(bundle) == TOP_LEVEL
    assert bundle["schema_version"] == SCHEMA_VERSION
    assert set(bundle["sweeps"]) == {"validation", "test"}
    served = bundle["meta"]["served_label"]
    assert served in bundle["pr_curves"]["models"]
    assert served in bundle["calibration"]
    assert "isolation_forest" not in bundle["calibration"]
    assert len(bundle["search"]) == 4


def test_bundle_written_next_to_served_model(bundle_run: tuple[Path, dict[str, Any]]) -> None:
    cfg_path, bundle = bundle_run
    cfg = load_config(cfg_path)
    copy = cfg.artifacts_dir / SELECTED / latest_version(cfg.artifacts_dir) / "dashboard.json"
    assert json.loads(copy.read_text(encoding="utf-8")) == bundle


def test_metrics_match_csv(bundle_run: tuple[Path, dict[str, Any]]) -> None:
    cfg_path, bundle = bundle_run
    csv = pd.read_csv(load_config(cfg_path).reports_dir / "metrics.csv")
    assert len(bundle["metrics"]) == len(csv)
    served = bundle["meta"]["served_label"]
    row = next(
        r
        for r in bundle["metrics"]
        if r["model"] == served and r["metric"] == "pr_auc" and r["split"] == "test"
    )
    assert row["ci_low"] <= row["value"] <= row["ci_high"]


def test_sweep_chosen_point_matches_report(bundle_run: tuple[Path, dict[str, Any]]) -> None:
    _, bundle = bundle_run
    test = bundle["sweeps"]["test"]
    i = test["chosen_index"]
    tp = next(
        r["value"]
        for r in bundle["metrics"]
        if r["model"] == bundle["meta"]["served_label"] and r["metric"] == "tp"
    )
    assert test["tp"][i] == tp
    for key in ("tp", "fp", "fn", "tn"):
        assert len(test[key]) == len(test["threshold"])
    # Thresholds are in decreasing order; counts move monotonically.
    assert all(np.diff(test["threshold"]) <= 0)
    assert all(np.diff(test["tp"]) >= 0)
    assert all(np.diff(test["fn"]) <= 0)
    assert test["tp"][0] + test["fp"][0] == 0


def test_no_dataset_rows_in_bundle(bundle_run: tuple[Path, dict[str, Any]]) -> None:
    """No transaction-shaped records (keyed by input columns) and no per-row arrays."""
    _, bundle = bundle_run
    slice_sizes = {part["rows"] for part in bundle["split"].values()}
    inputs = set(pd.read_csv(FIXTURE_PATH, nrows=1).columns)

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            assert not (inputs & set(node)), sorted(inputs & set(node))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            assert len(node) not in slice_sizes
            for value in node:
                walk(value)

    walk(bundle)


def test_sweep_costs_match_cost_at_threshold(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    y = (rng.random(5000) < 0.02).astype(int)
    scores = np.clip(rng.normal(0.1 + 0.5 * y, 0.15), 0, 1)
    amounts = rng.lognormal(3, 1, 5000)
    cfg = load_config(write_small_config(tmp_path))
    chosen = 0.5
    payload = sweep_payload(y, scores, amounts, cfg, chosen)
    c = cfg.costs
    for i in (0, 10, payload["chosen_index"], payload["min_cost_index"], len(payload["tp"]) - 1):
        total = payload["missed_cost"][i] + payload["fp"][i] * c.false_alarm
        total += payload["tp"][i] * c.caught_fraud
        expected = cost_at_threshold(y, scores, amounts, payload["threshold"][i], c).total_cost
        assert total == pytest.approx(expected, abs=0.05)
    chosen_outcome = cost_at_threshold(y, scores, amounts, chosen, c)
    assert payload["tp"][payload["chosen_index"]] == chosen_outcome.tp
    assert payload["fp"][payload["chosen_index"]] == chosen_outcome.fp
    totals = [
        payload["missed_cost"][i]
        + payload["fp"][i] * c.false_alarm
        + payload["tp"][i] * c.caught_fraud
        for i in range(len(payload["tp"]))
    ]
    assert payload["min_cost_index"] == int(np.argmin(totals))
    assert len(payload["tp"]) <= 650


def test_pr_curve_is_exact_at_corners() -> None:
    y = np.array([1, 0, 0, 1, 0, 0, 0, 1])
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
    curve = pr_curve(y, s)
    assert curve["recall"][0] == 1.0
    assert curve["recall"][-1] == 0.0
    assert set(np.round(curve["recall"], 4)) == {0.0, 0.3333, 0.6667, 1.0}
