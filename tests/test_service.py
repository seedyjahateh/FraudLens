from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fraudlens.artifacts import SELECTED, latest_version
from fraudlens.config import load_config
from fraudlens.data import FEATURE_COLUMNS
from fraudlens.evaluate import evaluate
from fraudlens.train import train
from service.app import JsonFormatter, create_app, resolve_model_dir
from tests.conftest import write_small_config


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory: pytest.TempPathFactory) -> Path:
    cfg_path = write_small_config(tmp_path_factory.mktemp("svc"))
    cfg = load_config(cfg_path)
    train(cfg, cfg_path.read_text(encoding="utf-8"))
    evaluate(cfg)
    return cfg.artifacts_dir


@pytest.fixture(scope="module")
def client(artifacts: Path) -> Iterator[TestClient]:
    model_dir = artifacts / SELECTED / latest_version(artifacts)
    with TestClient(create_app(model_dir)) as c:
        yield c


def _transaction(fixture_df: pd.DataFrame, i: int = 0) -> dict[str, Any]:
    return {c: float(fixture_df.iloc[i][c]) for c in FEATURE_COLUMNS}


def test_score_valid_request(client: TestClient, fixture_df: pd.DataFrame) -> None:
    resp = client.post("/score", json=_transaction(fixture_df))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["decision"] in {"flag", "allow"}
    assert body["decision"] == (
        "flag" if body["fraud_probability"] >= body["threshold"] else "allow"
    )
    assert len(body["top_features"]) == 5
    assert {f["feature"] for f in body["top_features"]} <= set(FEATURE_COLUMNS)
    assert resp.headers["x-request-id"] == body["request_id"]


def test_fraud_scores_higher_than_legit(client: TestClient, fixture_df: pd.DataFrame) -> None:
    fraud_rows = fixture_df.index[fixture_df["Class"] == 1][:10]
    legit_rows = fixture_df.index[fixture_df["Class"] == 0][:10]

    def mean_prob(rows: pd.Index) -> float:
        probs = [
            client.post("/score", json=_transaction(fixture_df, int(i))).json()["fraud_probability"]
            for i in rows
        ]
        return sum(probs) / len(probs)

    assert mean_prob(fraud_rows) > mean_prob(legit_rows)


def test_integers_are_accepted(client: TestClient, fixture_df: pd.DataFrame) -> None:
    payload = _transaction(fixture_df)
    payload["Amount"] = 10
    payload["Time"] = 0
    assert client.post("/score", json=payload).status_code == 200


def test_request_id_is_echoed(client: TestClient, fixture_df: pd.DataFrame) -> None:
    resp = client.post("/score", json=_transaction(fixture_df), headers={"x-request-id": "abc"})
    assert resp.json()["request_id"] == "abc"


@pytest.mark.parametrize(
    ("mutate", "field"),
    [
        (lambda p: p.pop("V14"), "V14"),
        (lambda p: p.update(V3="1.5"), "V3"),
        (lambda p: p.update(V3="abc"), "V3"),
        (lambda p: p.update(Amount=-1.0), "Amount"),
        (lambda p: p.update(Time=-5.0), "Time"),
        (lambda p: p.update(V1=None), "V1"),
        (lambda p: p.update(V2=[1.0]), "V2"),
        (lambda p: p.update(extra=1.0), "extra"),
    ],
)
def test_invalid_requests_return_422(
    client: TestClient, fixture_df: pd.DataFrame, mutate: Any, field: str
) -> None:
    payload = _transaction(fixture_df)
    mutate(payload)
    resp = client.post("/score", json=payload)
    assert resp.status_code == 422
    body = resp.json()
    assert body["detail"] == "invalid transaction"
    assert any(e["field"] == field for e in body["errors"]), body
    assert all(e["message"] for e in body["errors"])


def test_nan_rejected(client: TestClient, fixture_df: pd.DataFrame) -> None:
    payload = _transaction(fixture_df)
    payload["V5"] = float("nan")
    raw = json.dumps(payload)  # emits a bare NaN token
    assert "NaN" in raw
    resp = client.post("/score", content=raw, headers={"content-type": "application/json"})
    assert resp.status_code == 422
    assert [e["field"] for e in resp.json()["errors"]] == ["V5"]


def test_health(client: TestClient, artifacts: Path) -> None:
    body = client.get("/health").json()
    assert body == {
        "status": "ok",
        "model_loaded": True,
        "model_version": latest_version(artifacts),
    }


def test_model_reports_loaded_version(client: TestClient, artifacts: Path) -> None:
    body = client.get("/model").json()
    assert body["model_version"] == latest_version(artifacts)
    assert body["test_pr_auc"] is not None
    assert len(body["test_pr_auc_ci"]) == 2
    assert 0.0 < body["threshold"] < 1.0
    assert body["features"] == FEATURE_COLUMNS
    assert body["trained_at"] != "unknown"


def test_logs_contain_no_feature_values(
    client: TestClient, fixture_df: pd.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    payload = _transaction(fixture_df, 5)
    payload["Amount"] = 1234.5678
    payload["V7"] = -9.87654321
    caplog.set_level(logging.INFO, logger="fraudlens.service")
    resp = client.post("/score", json=payload, headers={"x-request-id": "log-check"})
    assert resp.status_code == 200
    formatter = JsonFormatter()
    lines = [formatter.format(r) for r in caplog.records if r.name == "fraudlens.service"]
    request_lines = [json.loads(line) for line in lines if '"log-check"' in line]
    assert len(request_lines) == 1
    entry = request_lines[0]
    assert entry["event"] == "request"
    assert entry["decision"] in {"flag", "allow"}
    assert entry["model_version"] == resp.json()["model_version"]
    assert entry["latency_ms"] >= 0
    text = "\n".join(lines)
    for value in ("1234.5678", "-9.87654321", "V7", "Amount"):
        assert value not in text


def test_resolve_model_dir(
    artifacts: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("FRAUDLENS_MODEL_DIR", raising=False)
    monkeypatch.setenv("FRAUDLENS_ARTIFACTS", str(artifacts))
    assert resolve_model_dir() == artifacts / SELECTED / latest_version(artifacts)
    monkeypatch.setenv("FRAUDLENS_MODEL_DIR", str(tmp_path))
    assert resolve_model_dir() == tmp_path


def test_app_from_environment(artifacts: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FRAUDLENS_MODEL_DIR", raising=False)
    monkeypatch.setenv("FRAUDLENS_ARTIFACTS", str(artifacts))
    with TestClient(create_app()) as c:
        assert c.get("/health").status_code == 200


def test_missing_artifacts_fail_at_startup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FRAUDLENS_MODEL_DIR", raising=False)
    monkeypatch.setenv("FRAUDLENS_ARTIFACTS", str(tmp_path))
    with pytest.raises(FileNotFoundError), TestClient(create_app()):
        pass


def test_wrong_artifact_type_rejected(tmp_path: Path) -> None:
    import joblib

    joblib.dump({"not": "a model"}, tmp_path / "model.joblib")
    with pytest.raises(TypeError), TestClient(create_app(tmp_path)):
        pass


def test_service_never_retrains(
    client: TestClient, fixture_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FR-18: scoring must not call fit on anything."""
    from sklearn.base import BaseEstimator

    def boom(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("fit called while serving")

    monkeypatch.setattr(BaseEstimator, "fit", boom, raising=False)
    assert client.post("/score", json=_transaction(fixture_df)).status_code == 200
