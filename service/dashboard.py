"""Dashboard support for the API: live stats, presets, the data bundle and the static SPA.

The dashboard is a static single-page app (``dashboard/``) built to plain files and served
here at ``/dashboard``. It talks to the same origin, so no CORS is needed.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse, Response

from fraudlens.artifacts import FraudModel

PRESETS_DIR = Path(__file__).with_name("presets")
WINDOW = 1000
RECENT = 120

# The SPA needs inline style attributes (chart libraries set them) but nothing else from
# outside this origin: fonts and scripts are bundled.
CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; font-src 'self'; connect-src 'self'; "
    "frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
)
SECURITY_HEADERS = {
    "Content-Security-Policy": CSP,
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
}


class ServiceStats:
    """Rolling latency and decision counts for ``/score`` (no feature values, ever)."""

    def __init__(self, window: int = WINDOW) -> None:
        self._lock = threading.Lock()
        self._latency: deque[float] = deque(maxlen=window)
        self._flags: deque[bool] = deque(maxlen=window)
        self.total = 0
        self.errors = 0
        self.started = time.time()

    def record(self, latency_ms: float, decision: str | None, status: int) -> None:
        with self._lock:
            self.total += 1
            if status != 200 or decision is None:
                self.errors += 1
                return
            self._latency.append(latency_ms)
            self._flags.append(decision == "flag")

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            lat = np.asarray(self._latency, dtype=np.float64)
            flags = np.asarray(self._flags, dtype=bool)
            total, errors = self.total, self.errors
        has = lat.size > 0
        return {
            "started_at": datetime.fromtimestamp(self.started, UTC).isoformat(timespec="seconds"),
            "uptime_seconds": round(time.time() - self.started, 1),
            "score_requests": total,
            "rejected_requests": errors,
            "window": int(lat.size),
            "p50_ms": float(np.percentile(lat, 50)) if has else None,
            "p95_ms": float(np.percentile(lat, 95)) if has else None,
            "p99_ms": float(np.percentile(lat, 99)) if has else None,
            "mean_ms": float(lat.mean()) if has else None,
            "flag_rate": float(flags.mean()) if has else None,
            "flagged": int(flags.sum()),
            "allowed": int(flags.size - flags.sum()),
            "recent_ms": [round(float(v), 3) for v in lat[-RECENT:]],
        }


def read_optional_json(path: Path) -> Any | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def presets(model: FraudModel) -> dict[str, dict[str, float]]:
    """Example transactions for the playground: no dataset rows are involved.

    ``typical`` is the training median of every input (the explanations' reference point);
    ``suspicious`` is a hand-made transaction with fraud-like components.
    """
    typical = {c: float(model.reference[c]) for c in model.feature_columns}
    suspicious = json.loads((PRESETS_DIR / "example_transaction.json").read_text(encoding="utf-8"))
    return {"typical": typical, "suspicious": {k: float(v) for k, v in suspicious.items()}}


def dashboard_dir() -> Path:
    configured = os.environ.get("FRAUDLENS_DASHBOARD_DIR")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[1] / "dashboard" / "dist"


def mount_dashboard(app: FastAPI, root: Path | None = None) -> bool:
    """Serve the built SPA at ``/dashboard`` with an ``index.html`` fallback for client
    routes. Returns False (and mounts nothing) when no build is present."""
    root = (root or dashboard_dir()).resolve()
    index = root / "index.html"
    if not index.is_file():
        return False

    def _file(path: Path, immutable: bool) -> FileResponse:
        cache = "public, max-age=31536000, immutable" if immutable else "no-cache"
        return FileResponse(path, headers={**SECURITY_HEADERS, "Cache-Control": cache})

    @app.get("/", include_in_schema=False)
    def root_redirect() -> RedirectResponse:
        return RedirectResponse("/dashboard/")

    @app.get("/dashboard", include_in_schema=False)
    def dashboard_redirect() -> RedirectResponse:
        return RedirectResponse("/dashboard/")

    @app.get("/dashboard/{path:path}", include_in_schema=False)
    def dashboard(path: str) -> Response:
        if path:
            candidate = (root / path).resolve()
            if not candidate.is_relative_to(root):
                raise HTTPException(status_code=404)
            if candidate.is_file():
                return _file(candidate, immutable=path.startswith("assets/"))
            if "." in Path(path).name:  # a missing asset, not a client-side route
                raise HTTPException(status_code=404)
        return _file(index, immutable=False)

    return True
