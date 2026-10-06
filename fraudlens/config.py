"""Typed access to ``config.yaml``."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SplitConfig:
    train: float = 0.6
    validation: float = 0.2
    test: float = 0.2

    def __post_init__(self) -> None:
        total = self.train + self.validation + self.test
        if abs(total - 1.0) > 1e-9 or min(self.train, self.validation, self.test) <= 0:
            raise ValueError(f"split ratios must be positive and sum to 1, got {total}")


@dataclass(frozen=True)
class CostConfig:
    missed_fraud: str | float = "amount"
    false_alarm: float = 5.0
    caught_fraud: float = 5.0

    def __post_init__(self) -> None:
        if isinstance(self.missed_fraud, str) and self.missed_fraud != "amount":
            raise ValueError("costs.missed_fraud must be 'amount' or a number")
        if self.false_alarm < 0 or self.caught_fraud < 0:
            raise ValueError("costs must be non-negative")


@dataclass(frozen=True)
class EvaluationConfig:
    bootstrap_samples: int = 1000
    confidence: float = 0.95
    recall_at_precision: float = 0.90
    precision_at_recall: float = 0.80
    calibration_bins: int = 10


@dataclass(frozen=True)
class ModelSearch:
    n_iter: int
    space: dict[str, list[Any]]


@dataclass(frozen=True)
class ImbalanceConfig:
    strategies: list[str] = field(default_factory=lambda: ["class_weight", "undersample", "smote"])
    models: list[str] = field(default_factory=list)
    undersample_ratio: float = 0.1
    smote_ratio: float = 0.1


@dataclass(frozen=True)
class Config:
    seed: int
    data_path: Path
    split: SplitConfig
    split_index_path: Path
    artifacts_dir: Path
    reports_dir: Path
    readme_path: Path
    costs: CostConfig
    evaluation: EvaluationConfig
    search: dict[str, ModelSearch]
    imbalance: ImbalanceConfig
    calibration_method: str = "sigmoid"
    permutation_repeats: int = 5
    top_k: int = 5


def load_config(path: str | Path = "config.yaml") -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    data = raw["data"]
    paths = raw.get("paths", {})
    search_raw = dict(raw["search"])
    search_raw.pop("metric", None)
    search = {
        name: ModelSearch(n_iter=int(spec["n_iter"]), space=dict(spec["space"]))
        for name, spec in search_raw.items()
    }
    method = raw.get("calibration", {}).get("method", "sigmoid")
    if method not in {"sigmoid", "isotonic"}:
        raise ValueError("calibration.method must be 'sigmoid' or 'isotonic'")
    explain = raw.get("explain", {})
    return Config(
        seed=int(raw["seed"]),
        data_path=Path(data["path"]),
        split=SplitConfig(**data["split"]),
        split_index_path=Path(data["split_index_path"]),
        artifacts_dir=Path(paths.get("artifacts", "artifacts")),
        reports_dir=Path(paths.get("reports", "reports")),
        readme_path=Path(paths.get("readme", "README.md")),
        costs=CostConfig(**raw["costs"]),
        evaluation=EvaluationConfig(**raw.get("evaluation", {})),
        search=search,
        imbalance=ImbalanceConfig(**raw.get("imbalance", {})),
        calibration_method=method,
        permutation_repeats=int(explain.get("permutation_repeats", 5)),
        top_k=int(explain.get("top_k", 5)),
    )
