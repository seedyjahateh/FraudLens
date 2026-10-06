"""Generate the small synthetic dataset used by tests and CI.

Same schema as the real file (Time, V1-V28, Amount, Class) with a few signal-bearing
components so models have something to learn. It is not a model of real fraud.

    uv run python tests/fixtures/make_fixture.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FIXTURE_PATH = Path(__file__).with_name("creditcard_sample.csv")
SIGNAL = {"V14": -3.0, "V4": 2.0, "V12": -2.0, "V10": -1.5, "V17": -2.0}


def make_synthetic(n: int = 3000, fraud_rate: float = 0.03, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = np.sort(rng.integers(0, 172_800, size=n)).astype(np.float64)
    y = (rng.random(n) < fraud_rate).astype(np.int64)
    data: dict[str, np.ndarray] = {"Time": times}
    for i in range(1, 29):
        col = rng.normal(0.0, 1.0, size=n)
        shift = SIGNAL.get(f"V{i}", 0.0)
        col = col + shift * y * rng.uniform(0.3, 1.2, size=n)
        data[f"V{i}"] = np.round(col, 4)
    amount = np.round(rng.lognormal(3.0, 1.2, size=n), 2)
    amount[y == 1] = np.round(amount[y == 1] * rng.uniform(0.5, 3.0, size=int(y.sum())), 2)
    data["Amount"] = amount
    data["Class"] = y
    return pd.DataFrame(data)


if __name__ == "__main__":
    df = make_synthetic()
    df.to_csv(FIXTURE_PATH, index=False, lineterminator="\n")
    print(f"wrote {FIXTURE_PATH} rows={len(df)} frauds={int(df['Class'].sum())}")
