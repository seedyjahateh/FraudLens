"""Load, validate and split the card-transaction dataset (FR-1, FR-2, EV-1)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fraudlens.config import SplitConfig

PCA_COLUMNS: list[str] = [f"V{i}" for i in range(1, 29)]
FEATURE_COLUMNS: list[str] = ["Time", *PCA_COLUMNS, "Amount"]
LABEL = "Class"
SCHEMA_COLUMNS: list[str] = [*FEATURE_COLUMNS, LABEL]


class SchemaError(ValueError):
    """The input file does not match the expected transaction schema."""


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Check columns, dtypes, missing values and labels. Raise ``SchemaError`` on mismatch."""
    missing = [c for c in SCHEMA_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in SCHEMA_COLUMNS]
    if missing or extra:
        raise SchemaError(f"column mismatch: missing={missing} unexpected={extra}")
    non_numeric = [c for c in SCHEMA_COLUMNS if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise SchemaError(f"non-numeric columns: {non_numeric}")
    nulls = df[SCHEMA_COLUMNS].isna().sum()
    if nulls.any():
        raise SchemaError(f"missing values: {nulls[nulls > 0].to_dict()}")
    labels = set(np.unique(df[LABEL]).tolist())
    if not labels <= {0, 1}:
        raise SchemaError(f"{LABEL} must be in {{0, 1}}, found {sorted(labels)}")
    if (df["Amount"] < 0).any():
        raise SchemaError("Amount must be non-negative")
    if (df["Time"] < 0).any():
        raise SchemaError("Time must be non-negative")
    out = df[SCHEMA_COLUMNS].copy()
    out[LABEL] = out[LABEL].astype(np.int64)
    return out


def load_transactions(path: str | Path) -> pd.DataFrame:
    """Read and validate the CSV. Fails loudly if anything is off."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. See data/README.md for how to obtain the dataset."
        )
    try:
        df = pd.read_csv(path)
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise SchemaError(f"could not parse {path}: {exc}") from exc
    return validate(df)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Split:
    """Row positions (into the validated frame) of each chronological slice."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray

    def to_json(self, path: str | Path, data_sha256: str) -> None:
        payload = {
            "data_sha256": data_sha256,
            "train": self.train.tolist(),
            "validation": self.validation.tolist(),
            "test": self.test.tolist(),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path, data_sha256: str | None = None) -> Split:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if data_sha256 is not None and payload["data_sha256"] != data_sha256:
            raise ValueError("split index file was built from different data; re-run training")
        return cls(
            train=np.asarray(payload["train"], dtype=np.int64),
            validation=np.asarray(payload["validation"], dtype=np.int64),
            test=np.asarray(payload["test"], dtype=np.int64),
        )


def _boundary(times: np.ndarray, cut: int) -> int:
    """Move a cut forward so that no timestamp straddles two slices."""
    n = len(times)
    while 0 < cut < n and times[cut] == times[cut - 1]:
        cut += 1
    return cut


def time_split(df: pd.DataFrame, ratios: SplitConfig) -> Split:
    """EV-1: sort by ``Time``; first 60% train, next 20% validation, last 20% test.

    Cuts are nudged forward to the next distinct timestamp so every validation row is
    strictly later than every training row, and every test row strictly later than
    every validation row.
    """
    order = np.argsort(df["Time"].to_numpy(), kind="mergesort")
    times = df["Time"].to_numpy()[order]
    n = len(order)
    cut1 = _boundary(times, round(n * ratios.train))
    cut2 = _boundary(times, max(cut1, round(n * (ratios.train + ratios.validation))))
    split = Split(train=order[:cut1], validation=order[cut1:cut2], test=order[cut2:])
    if min(len(split.train), len(split.validation), len(split.test)) == 0:
        raise ValueError("time split produced an empty slice")
    return split


def xy(df: pd.DataFrame, rows: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """Features (raw input columns) and labels for a set of row positions."""
    part = df.iloc[rows]
    return part[FEATURE_COLUMNS].reset_index(drop=True), part[LABEL].to_numpy()
