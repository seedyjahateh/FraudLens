"""NFR-2 load test: sequential POST /score requests, client-side latency percentiles.

    uv run python scripts/load_test.py --url http://localhost:8000 --environment "Docker ..."

Transactions are drawn from the test slice of the real dataset when it is available
(falling back to the synthetic fixture). Results go to ``reports/load_test.json`` and the
README latency section is regenerated from that file. The result is reported, not asserted.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np

from fraudlens.artifacts import SELECTED, write_json
from fraudlens.config import load_config
from fraudlens.data import FEATURE_COLUMNS, Split, file_sha256, load_transactions
from fraudlens.report import update_readme

FIXTURE = Path("tests/fixtures/creditcard_sample.csv")
EXAMPLE = Path(__file__).resolve().parents[1] / "service" / "presets" / "example_transaction.json"


def sample_transactions(config: Path, n: int, seed: int) -> tuple[list[dict[str, float]], str]:
    cfg = load_config(config)
    if cfg.data_path.exists() and cfg.split_index_path.exists():
        df = load_transactions(cfg.data_path)
        rows = Split.from_json(cfg.split_index_path, file_sha256(cfg.data_path)).test
        source = "test slice of the real dataset"
    else:
        df = load_transactions(FIXTURE)
        rows = np.arange(len(df))
        source = "synthetic fixture"
    picked = np.random.default_rng(seed).choice(rows, size=n, replace=True)
    frame = df.iloc[picked][FEATURE_COLUMNS]
    return [{k: float(v) for k, v in r.items()} for r in frame.to_dict("records")], source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--budget-ms", type=float, default=50.0)
    parser.add_argument("--environment", default=f"{platform.system()} {platform.machine()}")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--out", type=Path, default=Path("reports/load_test.json"))
    parser.add_argument("--no-readme", action="store_true")
    args = parser.parse_args()

    payloads, source = sample_transactions(args.config, args.requests + args.warmup, seed=0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    latencies: list[float] = []
    errors = 0
    flagged = 0
    with httpx.Client(base_url=args.url, timeout=10.0) as client:
        model = client.get("/model").json()
        example = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        response = client.post("/score", json=example, headers={"x-request-id": "example"})
        response.raise_for_status()
        write_json(args.out.parent / "example_response.json", response.json())
        for i, payload in enumerate(payloads):
            start = time.perf_counter()
            resp = client.post("/score", json=payload)
            elapsed = (time.perf_counter() - start) * 1000.0
            if i < args.warmup:
                continue
            latencies.append(elapsed)
            if resp.status_code != 200:
                errors += 1
            elif resp.json()["decision"] == "flag":
                flagged += 1

    lat = np.asarray(latencies)
    result = {
        "requests": len(lat),
        "warmup_requests": args.warmup,
        "target": args.url,
        "environment": args.environment,
        "payload_source": source,
        "model_version": model["model_version"],
        "p50_ms": float(np.percentile(lat, 50)),
        "p95_ms": float(np.percentile(lat, 95)),
        "p99_ms": float(np.percentile(lat, 99)),
        "mean_ms": float(lat.mean()),
        "max_ms": float(lat.max()),
        "errors": errors,
        "flagged": flagged,
        "budget_ms": args.budget_ms,
        "measured_at": datetime.now(UTC).strftime("%Y-%m-%d"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, result)
    print(json.dumps(result, indent=2))
    cfg = load_config(args.config)
    # Also keep it next to the served model, where the API's /api/dashboard picks it up.
    model_dir = cfg.artifacts_dir / SELECTED / str(model["model_version"])
    if model_dir.is_dir():
        write_json(model_dir / "load_test.json", result)
    if not args.no_readme:
        update_readme(cfg.readme_path, args.out.parent)


if __name__ == "__main__":
    main()
