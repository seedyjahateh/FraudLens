from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from fraudlens.config import Config, SplitConfig, load_config
from fraudlens.data import Split, load_transactions, time_split
from tests.fixtures.make_fixture import FIXTURE_PATH

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def fixture_df() -> pd.DataFrame:
    return load_transactions(FIXTURE_PATH)


@pytest.fixture(scope="session")
def fixture_split(fixture_df: pd.DataFrame) -> Split:
    return time_split(fixture_df, SplitConfig())


def write_small_config(tmp: Path) -> Path:
    """The repo config with tiny budgets and paths redirected into ``tmp``."""
    tmp.mkdir(parents=True, exist_ok=True)
    raw = yaml.safe_load((REPO_ROOT / "config.yaml").read_text(encoding="utf-8"))
    raw["data"]["path"] = str(FIXTURE_PATH)
    raw["data"]["split_index_path"] = str(tmp / "processed" / "split_indices.json")
    raw["paths"] = {
        "artifacts": str(tmp / "artifacts"),
        "reports": str(tmp / "reports"),
        "readme": str(tmp / "README.md"),
    }
    raw["evaluation"]["bootstrap_samples"] = 50
    raw["explain"]["permutation_repeats"] = 2
    for name, spec in raw["search"].items():
        if name == "metric":
            continue
        spec["n_iter"] = 2
    raw["search"]["random_forest"]["space"]["n_estimators"] = [20]
    raw["search"]["hist_gradient_boosting"]["space"]["max_iter"] = [50]
    raw["search"]["isolation_forest"]["space"]["n_estimators"] = [50]
    path = tmp / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    (tmp / "README.md").write_text(
        "# Test\n\n<!-- RESULTS:START -->\n<!-- RESULTS:END -->\n\n"
        "<!-- LATENCY:START -->\n<!-- LATENCY:END -->\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture()
def small_config(tmp_path: Path) -> Config:
    return load_config(write_small_config(tmp_path))
