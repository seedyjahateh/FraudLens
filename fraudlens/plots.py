"""Report figures. Rendered headless with matplotlib's Agg backend."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import StrMethodFormatter
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, precision_recall_curve

PALETTE = ["#2a6fdb", "#e0731f", "#2e9e5b", "#9b59b6", "#c0392b", "#7f8c8d"]
METADATA = {"Software": None}  # keep PNGs byte-stable across matplotlib builds


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130, metadata=METADATA)
    plt.close(fig)


def pr_curves(y: np.ndarray, scores: dict[str, np.ndarray], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for color, (name, s) in zip(PALETTE, scores.items(), strict=False):
        precision, recall, _ = precision_recall_curve(y, s)
        ap = average_precision_score(y, s)
        ax.step(recall, precision, where="post", color=color, lw=1.6, label=f"{name} (AP {ap:.3f})")
    prevalence = float(np.mean(y))
    ax.axhline(prevalence, color="#555", ls=":", lw=1, label=f"no skill ({prevalence:.4f})")
    ax.set(
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0, 1),
        ylim=(0, 1.02),
        title="Precision-recall on the held-out test slice",
    )
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    _save(fig, path)


def calibration(y: np.ndarray, scores: dict[str, np.ndarray], path: Path, bins: int = 10) -> None:
    """Reliability diagram (uniform bins, empty bins skipped) above a histogram of scores.

    With 0.17% fraud almost every score sits in the first bin, so the upper bins rest on
    few transactions; the histogram shows how many.
    """
    fig, (ax, hist) = plt.subplots(
        2, 1, figsize=(7, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]}
    )
    edges = np.linspace(0.0, 1.0, bins + 1)
    for color, (name, s) in zip(PALETTE, scores.items(), strict=False):
        obs, pred = calibration_curve(y, s, n_bins=bins, strategy="uniform")
        ax.plot(pred, obs, marker="o", ms=4, lw=1.4, color=color, label=name)
        hist.hist(s, bins=edges, histtype="step", lw=1.4, color=color)
    ax.plot([0, 1], [0, 1], color="#555", ls=":", lw=1, label="perfectly calibrated")
    ax.set(ylabel="Observed fraud rate", ylim=(-0.02, 1.02), title="Calibration on the test slice")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    hist.set(yscale="log", xlabel="Predicted fraud probability", ylabel="Transactions (log)")
    hist.grid(alpha=0.25)
    _save(fig, path)


def cost_curves(
    validation: tuple[np.ndarray, np.ndarray],
    test: tuple[np.ndarray, np.ndarray],
    threshold: float,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for (thresholds, totals), label, color in (
        (validation, "validation (threshold chosen here)", PALETTE[0]),
        (test, "test (reported once)", PALETTE[1]),
    ):
        t = np.clip(thresholds, 1e-6, 1.0)
        ax.plot(t, totals, color=color, lw=1.6, label=label)
    ax.axvline(threshold, color="#333", ls="--", lw=1, label=f"chosen threshold {threshold:.3g}")
    ax.set(
        xscale="log",
        xlabel="Decision threshold (flag if probability >= threshold, log)",
        ylabel="Total cost ($)",
        title="Total cost by threshold",
    )
    # Flagging almost everything costs far more than flagging nothing; cap the view just
    # above the flag-nothing cost so the region around the optimum is readable.
    ax.set_ylim(0, 1.15 * max(validation[1][0], test[1][0]))
    ax.yaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}"))
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, which="both")
    _save(fig, path)


def importance(frame: pd.DataFrame, path: Path, top: int = 15) -> None:
    head = frame.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(head["feature"], head["importance_mean"], xerr=head["importance_std"], color=PALETTE[0])
    ax.set(
        xlabel="Drop in validation PR-AUC when shuffled",
        title=f"Permutation importance (top {top})",
    )
    ax.grid(alpha=0.25, axis="x")
    _save(fig, path)
