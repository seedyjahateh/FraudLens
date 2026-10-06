from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraudlens.config import SplitConfig, load_config
from fraudlens.data import (
    FEATURE_COLUMNS,
    SchemaError,
    Split,
    file_sha256,
    load_transactions,
    time_split,
    validate,
    xy,
)
from tests.conftest import REPO_ROOT
from tests.fixtures.make_fixture import FIXTURE_PATH, make_synthetic


def test_fixture_matches_generator() -> None:
    on_disk = pd.read_csv(FIXTURE_PATH)
    pd.testing.assert_frame_equal(on_disk, make_synthetic(), check_exact=False, atol=1e-6)


def test_load_valid_fixture(fixture_df: pd.DataFrame) -> None:
    assert len(fixture_df) == 3000
    assert set(fixture_df["Class"].unique()) == {0, 1}


def test_missing_file_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"data/README\.md"):
        load_transactions(tmp_path / "nope.csv")


def test_unparseable_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_bytes(b"\xff\xfe\x00garbage")
    with pytest.raises(SchemaError):
        load_transactions(bad)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d.drop(columns=["V3"]), "missing"),
        (lambda d: d.assign(extra=1), "unexpected"),
        (lambda d: d.assign(V5="x"), "non-numeric"),
        (lambda d: d.assign(V7=np.where(d.index == 3, np.nan, d["V7"])), "missing values"),
        (lambda d: d.assign(Class=np.where(d.index == 0, 2, d["Class"])), "Class"),
        (lambda d: d.assign(Amount=-d["Amount"] - 1), "Amount"),
        (lambda d: d.assign(Time=d["Time"] - 1e6), "Time"),
    ],
)
def test_schema_rejects_bad_frames(fixture_df: pd.DataFrame, mutate: object, message: str) -> None:
    bad = mutate(fixture_df.copy())  # type: ignore[operator]
    with pytest.raises(SchemaError, match=message):
        validate(bad)


def test_schema_rejects_bad_csv(tmp_path: Path, fixture_df: pd.DataFrame) -> None:
    path = tmp_path / "bad.csv"
    fixture_df.drop(columns=["Amount"]).to_csv(path, index=False)
    with pytest.raises(SchemaError):
        load_transactions(path)


def test_time_split_is_chronological(fixture_df: pd.DataFrame, fixture_split: Split) -> None:
    t = fixture_df["Time"].to_numpy()
    assert t[fixture_split.train].max() < t[fixture_split.validation].min()
    assert t[fixture_split.validation].max() < t[fixture_split.test].min()
    assert t[fixture_split.train].max() < t[fixture_split.test].min()


def test_time_split_partitions_rows(fixture_df: pd.DataFrame, fixture_split: Split) -> None:
    allrows = np.concatenate([fixture_split.train, fixture_split.validation, fixture_split.test])
    assert sorted(allrows.tolist()) == list(range(len(fixture_df)))
    n = len(fixture_df)
    assert abs(len(fixture_split.train) / n - 0.6) < 0.02
    assert abs(len(fixture_split.test) / n - 0.2) < 0.02


def test_time_split_handles_ties_and_shuffled_input(fixture_df: pd.DataFrame) -> None:
    df = fixture_df.copy()
    df["Time"] = (df["Time"] // 1000) * 1000  # many ties
    df = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    split = time_split(df, SplitConfig())
    t = df["Time"].to_numpy()
    assert t[split.train].max() < t[split.validation].min()
    assert t[split.validation].max() < t[split.test].min()


def test_time_split_rejects_degenerate(fixture_df: pd.DataFrame) -> None:
    df = fixture_df.assign(Time=0.0)
    with pytest.raises(ValueError, match="empty"):
        time_split(df, SplitConfig())


def test_split_roundtrip(tmp_path: Path, fixture_split: Split) -> None:
    path = tmp_path / "split.json"
    fixture_split.to_json(path, "abc")
    loaded = Split.from_json(path, "abc")
    np.testing.assert_array_equal(loaded.test, fixture_split.test)
    with pytest.raises(ValueError, match="different data"):
        Split.from_json(path, "other")


def test_split_is_reproducible(fixture_df: pd.DataFrame, fixture_split: Split) -> None:
    again = time_split(fixture_df, SplitConfig())
    np.testing.assert_array_equal(again.train, fixture_split.train)


def test_xy(fixture_df: pd.DataFrame, fixture_split: Split) -> None:
    X, y = xy(fixture_df, fixture_split.validation)
    assert list(X.columns) == FEATURE_COLUMNS
    assert len(X) == len(y) == len(fixture_split.validation)


def test_sha256_stable() -> None:
    assert file_sha256(FIXTURE_PATH) == file_sha256(FIXTURE_PATH)
    assert len(file_sha256(FIXTURE_PATH)) == 64


def test_bad_split_ratios() -> None:
    with pytest.raises(ValueError):
        SplitConfig(train=0.7, validation=0.2, test=0.2)


def test_repo_config_loads() -> None:
    cfg = load_config(REPO_ROOT / "config.yaml")
    assert cfg.split.train == 0.6
    assert cfg.evaluation.bootstrap_samples == 1000
    assert cfg.costs.missed_fraud == "amount"
