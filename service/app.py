"""FastAPI scoring service (FR-15..FR-19).

The app loads one versioned artifact at startup and never trains. Which artifact:

- ``FRAUDLENS_MODEL_DIR``: an explicit ``artifacts/selected/<version>`` directory, or
- ``FRAUDLENS_ARTIFACTS``: an artifacts root; the version named in its ``LATEST`` file.

Run locally (``--factory`` so importing this module never needs trained artifacts)::

    FRAUDLENS_ARTIFACTS=artifacts uvicorn --factory service.app:create_app --port 8000
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from fraudlens.artifacts import SELECTED, FraudModel, latest_version, load_selected
from fraudlens.features import transaction_frame
from service.dashboard import ServiceStats, mount_dashboard, presets, read_optional_json
from service.schemas import (
    FeatureContribution,
    HealthResponse,
    ModelInfo,
    PresetsResponse,
    ScoreResponse,
    StatsResponse,
    Transaction,
)

TOP_K = 5
log = logging.getLogger("fraudlens.service")


class JsonFormatter(logging.Formatter):
    """One JSON object per line. Only fields passed in ``extra={"fields": ...}`` are
    emitted, so request bodies (feature values) can never leak into logs by accident."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "event": record.getMessage(),
        }
        payload.update(getattr(record, "fields", {}))
        return json.dumps(payload)


def configure_logging() -> None:
    if any(isinstance(h.formatter, JsonFormatter) for h in log.handlers):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = True


def resolve_model_dir() -> Path:
    explicit = os.environ.get("FRAUDLENS_MODEL_DIR")
    if explicit:
        return Path(explicit)
    root = Path(os.environ.get("FRAUDLENS_ARTIFACTS", "artifacts"))
    return root / SELECTED / latest_version(root)


def create_app(model_dir: Path | None = None, dashboard_root: Path | None = None) -> FastAPI:
    configure_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        directory = model_dir or resolve_model_dir()
        model = load_selected(directory)
        app.state.model = model
        app.state.model_dir = directory
        app.state.bundle = read_optional_json(directory / "dashboard.json")
        log.info(
            "model_loaded",
            extra={
                "fields": {
                    "model_version": model.version,
                    "model_name": model.model_name,
                    "path": str(directory),
                }
            },
        )
        yield

    app = FastAPI(
        title="FraudLens scoring API",
        version="0.1.0",
        description="Scores card transactions with a model trained on the public ULB dataset. "
        "A portfolio project, not a production system.",
        lifespan=lifespan,
    )
    app.state.stats = ServiceStats()

    @app.middleware("http")
    async def request_log(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = request_id
        start = time.perf_counter()
        response = await call_next(request)
        latency_ms = (time.perf_counter() - start) * 1000.0
        response.headers["x-request-id"] = request_id
        decision = getattr(request.state, "decision", None)
        if request.url.path == "/score" and request.method == "POST":
            request.app.state.stats.record(latency_ms, decision, response.status_code)
        model: FraudModel | None = getattr(request.app.state, "model", None)
        log.info(
            "request",
            extra={
                "fields": {
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "latency_ms": round(latency_ms, 3),
                    "decision": decision,
                    "model_version": model.version if model else None,
                }
            },
        )
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Report which field failed and why, without echoing submitted values.
        errors = [
            {
                "field": ".".join(str(p) for p in err["loc"] if p != "body"),
                "message": err["msg"],
                "type": err["type"],
            }
            for err in exc.errors()
        ]
        return JSONResponse(
            status_code=422, content={"detail": "invalid transaction", "errors": errors}
        )

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        model: FraudModel = request.app.state.model
        return HealthResponse(status="ok", model_loaded=True, model_version=model.version)

    @app.get("/model", response_model=ModelInfo)
    def model_info(request: Request) -> ModelInfo:
        model: FraudModel = request.app.state.model
        meta = model.metadata
        evaluation = meta.get("evaluation", {})
        return ModelInfo(
            model_name=model.model_name,
            model_version=model.version,
            trained_at=meta.get("trained_at", "unknown"),
            calibration=meta.get("calibration"),
            threshold=model.threshold,
            test_pr_auc=evaluation.get("test_pr_auc"),
            test_pr_auc_ci=evaluation.get("test_pr_auc_ci"),
            data_sha256=meta.get("data_sha256", "unknown"),
            code_version=meta.get("code_version", "unknown"),
            features=model.feature_columns,
        )

    @app.post("/score", response_model=ScoreResponse)
    def score(transaction: Transaction, request: Request) -> ScoreResponse:
        model: FraudModel = request.app.state.model
        row = transaction_frame(transaction.model_dump())
        probability, contributions = model.score(row, TOP_K)
        decision = model.decide(probability)
        request.state.decision = decision
        return ScoreResponse(
            request_id=request.state.request_id,
            fraud_probability=probability,
            decision=decision,
            threshold=model.threshold,
            model_version=model.version,
            top_features=[
                FeatureContribution(feature=c.feature, contribution=c.contribution)
                for c in contributions
            ],
        )

    @app.get("/api/dashboard", tags=["dashboard"])
    def dashboard_bundle(request: Request) -> JSONResponse:
        """Metrics, curves and sweeps for the loaded model version (written by evaluate)."""
        bundle = request.app.state.bundle
        if bundle is None:
            model: FraudModel = request.app.state.model
            return JSONResponse(
                status_code=404,
                content={
                    "detail": f"no dashboard bundle for model version {model.version}; "
                    "run `python -m fraudlens.evaluate`"
                },
            )
        load_test = read_optional_json(request.app.state.model_dir / "load_test.json")
        return JSONResponse({**bundle, "load_test": load_test})

    @app.get("/api/presets", response_model=PresetsResponse, tags=["dashboard"])
    def preset_transactions(request: Request) -> PresetsResponse:
        found = presets(request.app.state.model)
        return PresetsResponse(
            typical=Transaction(**found["typical"]),
            suspicious=Transaction(**found["suspicious"]),
        )

    @app.get("/api/stats", response_model=StatsResponse, tags=["dashboard"])
    def service_stats(request: Request) -> StatsResponse:
        return StatsResponse(**request.app.state.stats.snapshot())

    mount_dashboard(app, dashboard_root)
    return app
