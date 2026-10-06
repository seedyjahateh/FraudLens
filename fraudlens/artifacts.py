"""Versioned model artifacts (FR-10, FR-18).

Layout::

    artifacts/
      LATEST                          version of the most recent training run
      runs/<version>/summary.json     search trials, imbalance comparison, selection
      <model>/<version>/model.joblib  fitted pipeline for each candidate
      <model>/<version>/metadata.json
      selected/<version>/model.joblib calibrated chosen model + threshold (what the API serves)
      selected/<version>/metadata.json
      selected/<version>/evaluation.json  test metrics, written once by evaluate
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd

from fraudlens.explain import Contribution, explain_row
from fraudlens.models import fraud_scores

SELECTED = "selected"


@dataclass
class FraudModel:
    """Everything the scoring service needs, in one object."""

    model_name: str
    version: str
    estimator: Any
    threshold: float
    reference: dict[str, float]
    feature_columns: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return fraud_scores(self.estimator, X[self.feature_columns])

    def score(self, row: pd.DataFrame, top_k: int = 5) -> tuple[float, list[Contribution]]:
        return explain_row(self.estimator, row, self.reference, top_k)

    def decide(self, probability: float) -> Literal["flag", "allow"]:
        return "flag" if probability >= self.threshold else "allow"


def code_version() -> str:
    """The git commit the code was run from, marked ``-dirty`` with uncommitted changes."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


def run_version(data_sha256: str, config_text: str) -> str:
    """Deterministic version id: the same data and config give the same version."""
    config_sha = hashlib.sha256(config_text.encode("utf-8")).hexdigest()
    return f"{data_sha256[:8]}-{config_sha[:8]}"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def write_json(path: Path, payload: Any, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        json.dumps(payload, separators=(",", ":"), default=_json_default)
        if compact
        else json.dumps(payload, indent=2, default=_json_default)
    )
    path.write_text(text + "\n", encoding="utf-8", newline="\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_model(root: Path, name: str, version: str, model: Any, metadata: dict[str, Any]) -> Path:
    out = root / name / version
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model.joblib")
    write_json(out / "metadata.json", metadata)
    return out


def load_model(root: Path, name: str, version: str) -> tuple[Any, dict[str, Any]]:
    out = root / name / version
    return joblib.load(out / "model.joblib"), read_json(out / "metadata.json")


def latest_version(root: Path) -> str:
    pointer = root / "LATEST"
    if not pointer.exists():
        raise FileNotFoundError(f"no trained artifacts in {root}; run `python -m fraudlens.train`")
    return pointer.read_text(encoding="utf-8").strip()


def load_selected(artifact_dir: Path) -> FraudModel:
    """Load one versioned served model, e.g. ``artifacts/selected/<version>``."""
    model = joblib.load(artifact_dir / "model.joblib")
    if not isinstance(model, FraudModel):
        raise TypeError(f"{artifact_dir} does not contain a FraudModel")
    evaluation = artifact_dir / "evaluation.json"
    if evaluation.exists():
        model.metadata["evaluation"] = read_json(evaluation)
    return model
