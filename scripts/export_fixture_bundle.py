"""Regenerate the dashboard's test bundle from the synthetic fixture.

    uv run python scripts/export_fixture_bundle.py

Trains and evaluates with the tiny test configuration (synthetic data only) and copies the
resulting dashboard.json to dashboard/src/test/fixture-bundle.json, which the frontend
tests use to check their cost maths against numbers Python computed.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fraudlens.config import load_config
from fraudlens.evaluate import evaluate
from fraudlens.train import train
from tests.conftest import write_small_config

TARGET = Path(__file__).resolve().parents[1] / "dashboard" / "src" / "test" / "fixture-bundle.json"


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = write_small_config(Path(tmp))
        cfg = load_config(cfg_path)
        train(cfg, cfg_path.read_text(encoding="utf-8"))
        evaluate(cfg)
        shutil.copy(cfg.reports_dir / "dashboard.json", TARGET)
    print(f"wrote {TARGET}")


if __name__ == "__main__":
    main()
