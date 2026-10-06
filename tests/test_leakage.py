"""EV-6 leakage checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from fraudlens.artifacts import SELECTED, FraudModel, latest_version, load_selected
from fraudlens.config import ImbalanceConfig, load_config
from fraudlens.data import Split, xy
from fraudlens.features import build_features
from fraudlens.models import LOGISTIC, build_pipeline
from fraudlens.train import train
from tests.conftest import write_small_config


def test_scaler_statistics_come_from_training_rows_only(
    fixture_df: pd.DataFrame, fixture_split: Split
) -> None:
    X_tr, y_tr = xy(fixture_df, fixture_split.train)
    pipe = build_pipeline(LOGISTIC, {"sampling": "class_weight"}, seed=0).fit(X_tr, y_tr)
    scaler = pipe.named_steps["scale"]
    expected = build_features(X_tr)
    np.testing.assert_allclose(scaler.mean_, expected.mean().to_numpy())
    np.testing.assert_allclose(scaler.var_, expected.var(ddof=0).to_numpy())
    assert scaler.n_samples_seen_ == len(X_tr)


@pytest.mark.parametrize(
    ("sampling", "cls"), [("undersample", RandomUnderSampler), ("smote", SMOTE)]
)
def test_resampling_happens_in_fit_only(
    sampling: str,
    cls: Any,
    fixture_df: pd.DataFrame,
    fixture_split: Split,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Samplers see exactly the training rows once, and never run at predict time."""
    seen: list[int] = []
    original = cls.fit_resample

    def spy(self: Any, X: Any, y: Any, **kw: Any) -> Any:
        seen.append(len(X))
        return original(self, X, y, **kw)

    monkeypatch.setattr(cls, "fit_resample", spy)
    X_tr, y_tr = xy(fixture_df, fixture_split.train)
    X_va, _ = xy(fixture_df, fixture_split.validation)
    X_te, _ = xy(fixture_df, fixture_split.test)
    pipe = build_pipeline(LOGISTIC, {"sampling": sampling}, 0, ImbalanceConfig()).fit(X_tr, y_tr)
    pipe.predict_proba(X_va)
    pipe.predict_proba(X_te)
    assert seen == [len(X_tr)]


def _train_and_load(cfg_path: Path) -> FraudModel:
    cfg = load_config(cfg_path)
    train(cfg, cfg_path.read_text(encoding="utf-8"))
    return load_selected(cfg.artifacts_dir / SELECTED / latest_version(cfg.artifacts_dir))


def test_test_rows_never_influence_training(
    tmp_path: Path, fixture_df: pd.DataFrame, fixture_split: Split
) -> None:
    """Corrupt every test-slice feature and label: training must produce the same model."""
    clean = _train_and_load(write_small_config(tmp_path / "clean"))

    corrupted = fixture_df.copy()
    rows = corrupted.index[fixture_split.test]
    pca = [c for c in corrupted.columns if c.startswith("V")]
    corrupted.loc[rows, pca] = 1e6
    corrupted.loc[rows, "Amount"] = 1e7
    corrupted.loc[rows, "Class"] = 1 - corrupted.loc[rows, "Class"]
    data_path = tmp_path / "corrupted.csv"
    corrupted.to_csv(data_path, index=False)
    dirty_cfg = write_small_config(tmp_path / "dirty")
    raw = yaml.safe_load(dirty_cfg.read_text(encoding="utf-8"))
    raw["data"]["path"] = str(data_path)
    dirty_cfg.write_text(yaml.safe_dump(raw), encoding="utf-8")
    dirty = _train_and_load(dirty_cfg)

    assert clean.model_name == dirty.model_name
    assert clean.threshold == dirty.threshold
    assert clean.reference == dirty.reference
    X_va, _ = xy(fixture_df, fixture_split.validation)
    np.testing.assert_array_equal(clean.predict_proba(X_va), dirty.predict_proba(X_va))
