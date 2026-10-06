from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraudlens.data import FEATURE_COLUMNS
from fraudlens.features import (
    ENGINEERED_COLUMNS,
    FeatureBuilder,
    build_features,
    transaction_frame,
)


def test_engineered_columns(fixture_df: pd.DataFrame) -> None:
    out = build_features(fixture_df[FEATURE_COLUMNS])
    assert list(out.columns) == ENGINEERED_COLUMNS
    assert out["hour"].between(0, 23).all()
    np.testing.assert_allclose(out["log_amount"], np.log1p(fixture_df["Amount"]))


def test_hour_of_day() -> None:
    row = {c: 0.0 for c in FEATURE_COLUMNS}
    row["Time"] = 3600 * 25 + 10  # one day and one hour in
    assert build_features(transaction_frame(row))["hour"].iloc[0] == 1


def test_training_and_serving_paths_agree(fixture_df: pd.DataFrame) -> None:
    """The service builds a one-row frame from JSON; it must equal the batch path."""
    X = fixture_df[FEATURE_COLUMNS]
    batch = build_features(X)
    for i in [0, 17, 1234, len(X) - 1]:
        payload = {str(k): float(v) for k, v in X.iloc[i].to_dict().items()}
        single = FeatureBuilder().fit_transform(transaction_frame(payload))
        np.testing.assert_allclose(single.to_numpy()[0], batch.to_numpy()[i])


def test_missing_column_rejected(fixture_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="missing feature columns"):
        build_features(fixture_df.drop(columns=["V2"]))


def test_feature_names_out() -> None:
    assert list(FeatureBuilder().get_feature_names_out()) == ENGINEERED_COLUMNS
