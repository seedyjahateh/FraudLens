"""Dashboard endpoints: /api/dashboard, /api/presets, /api/stats and the static SPA mount."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fraudlens.artifacts import SELECTED, latest_version, write_json
from fraudlens.config import load_config
from fraudlens.data import FEATURE_COLUMNS
from fraudlens.evaluate import evaluate
from fraudlens.train import train
from service.app import create_app
from service.dashboard import ServiceStats, dashboard_dir
from tests.conftest import write_small_config


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    cfg_path = write_small_config(tmp_path_factory.mktemp("svcdash"))
    cfg = load_config(cfg_path)
    train(cfg, cfg_path.read_text(encoding="utf-8"))
    evaluate(cfg)
    return cfg.artifacts_dir / SELECTED / latest_version(cfg.artifacts_dir)


@pytest.fixture(scope="module")
def spa(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("dist")
    (root / "assets").mkdir()
    (root / "index.html").write_text("<!doctype html><title>FraudLens</title>", encoding="utf-8")
    (root / "assets" / "app-123.js").write_text("console.log(1)", encoding="utf-8")
    (root / "favicon.svg").write_text("<svg/>", encoding="utf-8")
    return root


@pytest.fixture()
def client(model_dir: Path, spa: Path) -> Iterator[TestClient]:
    with TestClient(create_app(model_dir, dashboard_root=spa)) as c:
        yield c


def _transaction(fixture_df: pd.DataFrame, i: int = 0) -> dict[str, Any]:
    return {c: float(fixture_df.iloc[i][c]) for c in FEATURE_COLUMNS}


def test_dashboard_bundle(client: TestClient, model_dir: Path) -> None:
    resp = client.get("/api/dashboard")
    assert resp.status_code == 200
    body = resp.json()
    assert body["schema_version"] == 1
    assert body["meta"]["version"] == model_dir.name
    assert body["load_test"] is None


def test_dashboard_bundle_includes_load_test(model_dir: Path, tmp_path: Path) -> None:
    copy = tmp_path / model_dir.name
    shutil.copytree(model_dir, copy)
    write_json(copy / "load_test.json", {"p95_ms": 12.3})
    with TestClient(create_app(copy)) as c:
        assert c.get("/api/dashboard").json()["load_test"] == {"p95_ms": 12.3}


def test_dashboard_bundle_missing(model_dir: Path, tmp_path: Path) -> None:
    copy = tmp_path / model_dir.name
    shutil.copytree(model_dir, copy)
    (copy / "dashboard.json").unlink()
    with TestClient(create_app(copy)) as c:
        resp = c.get("/api/dashboard")
    assert resp.status_code == 404
    assert "fraudlens.evaluate" in resp.json()["detail"]


def test_presets_are_valid_transactions(client: TestClient) -> None:
    body = client.get("/api/presets").json()
    assert set(body) == {"typical", "suspicious"}
    for preset in body.values():
        assert list(preset) == FEATURE_COLUMNS
        resp = client.post("/score", json=preset)
        assert resp.status_code == 200
    typical = client.post("/score", json=body["typical"]).json()["fraud_probability"]
    suspicious = client.post("/score", json=body["suspicious"]).json()["fraud_probability"]
    assert suspicious > typical


def test_stats_track_scores_and_rejections(client: TestClient, fixture_df: pd.DataFrame) -> None:
    empty = client.get("/api/stats").json()
    assert empty["score_requests"] == 0
    assert empty["p95_ms"] is None
    for i in range(5):
        assert client.post("/score", json=_transaction(fixture_df, i)).status_code == 200
    bad = _transaction(fixture_df)
    bad["Amount"] = -1
    assert client.post("/score", json=bad).status_code == 422
    client.get("/health")  # not counted
    stats = client.get("/api/stats").json()
    assert stats["score_requests"] == 6
    assert stats["rejected_requests"] == 1
    assert stats["window"] == 5
    assert stats["flagged"] + stats["allowed"] == 5
    assert len(stats["recent_ms"]) == 5
    assert 0 < stats["p50_ms"] <= stats["p95_ms"] <= stats["p99_ms"]
    assert not set(FEATURE_COLUMNS) & set(stats)


def test_stats_window_is_bounded() -> None:
    stats = ServiceStats(window=3)
    for i in range(10):
        stats.record(float(i), "flag" if i % 2 else "allow", 200)
    snap = stats.snapshot()
    assert snap["window"] == 3
    assert snap["recent_ms"] == [7.0, 8.0, 9.0]
    assert snap["score_requests"] == 10


def test_spa_index_and_security_headers(client: TestClient) -> None:
    resp = client.get("/dashboard/")
    assert resp.status_code == 200
    assert "FraudLens" in resp.text
    assert "default-src 'self'" in resp.headers["content-security-policy"]
    assert resp.headers["x-content-type-options"] == "nosniff"
    assert resp.headers["cache-control"] == "no-cache"


def test_spa_client_routes_fall_back_to_index(client: TestClient) -> None:
    for route in ("threshold", "playground", "model", "deep/nested/route"):
        resp = client.get(f"/dashboard/{route}")
        assert resp.status_code == 200
        assert "FraudLens" in resp.text


def test_spa_assets(client: TestClient) -> None:
    asset = client.get("/dashboard/assets/app-123.js")
    assert asset.status_code == 200
    assert "immutable" in asset.headers["cache-control"]
    assert client.get("/dashboard/favicon.svg").status_code == 200
    assert client.get("/dashboard/assets/missing.js").status_code == 404
    assert client.get("/dashboard/..%2F..%2Fsecret.txt").status_code == 404


def test_root_redirects_to_dashboard(client: TestClient) -> None:
    for path in ("/", "/dashboard"):
        resp = client.get(path, follow_redirects=False)
        assert resp.status_code in (302, 307)
        assert resp.headers["location"] == "/dashboard/"


def test_no_build_means_no_mount(model_dir: Path, tmp_path: Path) -> None:
    with TestClient(create_app(model_dir, dashboard_root=tmp_path)) as c:
        assert c.get("/dashboard/").status_code == 404
        assert c.get("/health").status_code == 200


def test_dashboard_dir_from_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FRAUDLENS_DASHBOARD_DIR", str(tmp_path))
    assert dashboard_dir() == tmp_path
    monkeypatch.delenv("FRAUDLENS_DASHBOARD_DIR")
    assert dashboard_dir().parts[-2:] == ("dashboard", "dist")
