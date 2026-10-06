"""Request and response models for the scoring API (FR-15, FR-16).

Requests are validated strictly: every model feature is required, values must be JSON
numbers (strings such as ``"1.5"`` are rejected rather than coerced), NaN/infinity are
rejected, unknown fields are rejected, and ``Amount`` and ``Time`` must be non-negative.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

Component = Annotated[float, Field(description="Anonymised PCA component")]

_STRICT = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)


class Transaction(BaseModel):
    """One card transaction with every model input."""

    model_config = ConfigDict(
        **_STRICT,
        json_schema_extra={
            "example": {
                "Time": 406.0,
                **{f"V{i}": 0.0 for i in range(1, 29)},
                "Amount": 149.62,
            }
        },
    )

    Time: float = Field(ge=0, description="Seconds since the first transaction in the dataset")
    V1: Component
    V2: Component
    V3: Component
    V4: Component
    V5: Component
    V6: Component
    V7: Component
    V8: Component
    V9: Component
    V10: Component
    V11: Component
    V12: Component
    V13: Component
    V14: Component
    V15: Component
    V16: Component
    V17: Component
    V18: Component
    V19: Component
    V20: Component
    V21: Component
    V22: Component
    V23: Component
    V24: Component
    V25: Component
    V26: Component
    V27: Component
    V28: Component
    Amount: float = Field(ge=0, description="Transaction amount; must be non-negative")


class FeatureContribution(BaseModel):
    feature: str
    contribution: float = Field(
        description="Fraud probability minus the probability with this feature set to its "
        "training median. Positive values push towards fraud."
    )


class ScoreResponse(BaseModel):
    request_id: str
    fraud_probability: float = Field(ge=0, le=1)
    decision: Literal["flag", "allow"]
    threshold: float
    model_version: str
    top_features: list[FeatureContribution]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
    model_version: str


class PresetsResponse(BaseModel):
    typical: Transaction = Field(description="Training median of every input")
    suspicious: Transaction = Field(description="Hand-made transaction, not from the dataset")


class StatsResponse(BaseModel):
    """Rolling statistics over the most recent ``/score`` calls served by this process."""

    started_at: str
    uptime_seconds: float
    score_requests: int
    rejected_requests: int
    window: int
    p50_ms: float | None
    p95_ms: float | None
    p99_ms: float | None
    mean_ms: float | None
    flag_rate: float | None
    flagged: int
    allowed: int
    recent_ms: list[float]


class ModelInfo(BaseModel):
    model_name: str
    model_version: str
    trained_at: str
    calibration: str | None
    threshold: float
    test_pr_auc: float | None = Field(
        description="PR-AUC on the held-out test slice; null until evaluation has run"
    )
    test_pr_auc_ci: list[float] | None
    data_sha256: str
    code_version: str
    features: list[str]
