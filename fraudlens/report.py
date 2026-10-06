"""Generate ``reports/report.md`` and the README results section (FR-11, FR-12).

The README section between ``<!-- RESULTS:START -->`` and ``<!-- RESULTS:END -->`` (and the
latency section between ``<!-- LATENCY:* -->`` markers) is rebuilt from the files in
``reports/``; it is never edited by hand.

    python -m fraudlens.report        # refresh the README from reports/
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from fraudlens.config import Config, load_config

RESULTS = ("<!-- RESULTS:START -->", "<!-- RESULTS:END -->")
LATENCY = ("<!-- LATENCY:START -->", "<!-- LATENCY:END -->")
EXAMPLE = ("<!-- EXAMPLE:START -->", "<!-- EXAMPLE:END -->")


def _num(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


def _money(x: float) -> str:
    return f"${x:,.0f}"


def _ci(metrics: pd.DataFrame, model: str, metric: str, digits: int = 3) -> str:
    row = metrics[
        (metrics["model"] == model) & (metrics["metric"] == metric) & (metrics["split"] == "test")
    ]
    if row.empty:
        return "n/a"
    r = row.iloc[0]
    if pd.isna(r["ci_low"]):
        return _num(r["value"], digits)
    return f"{_num(r['value'], digits)} ({_num(r['ci_low'], digits)}–{_num(r['ci_high'], digits)})"


def _pct_ci(metrics: pd.DataFrame, model: str, metric: str) -> str:
    row = metrics[
        (metrics["model"] == model) & (metrics["metric"] == metric) & (metrics["split"] == "test")
    ].iloc[0]
    return f"{row['value']:.1f}% (CI {row['ci_low']:.1f}–{row['ci_high']:.1f}%)"


def _value(metrics: pd.DataFrame, model: str, metric: str, split: str = "test") -> float:
    row = metrics[
        (metrics["model"] == model) & (metrics["metric"] == metric) & (metrics["split"] == split)
    ]
    if row.empty:
        raise KeyError(f"{model}/{split}/{metric} not in metrics.csv")
    return float(row.iloc[0]["value"])


def _metric_names(metrics: pd.DataFrame) -> tuple[str, str]:
    names = metrics["metric"].unique()
    r_at_p = next(m for m in names if m.startswith("recall_at_") and m.endswith("precision"))
    p_at_r = next(m for m in names if m.startswith("precision_at_") and m.endswith("recall"))
    return r_at_p, p_at_r


def _pct_label(metric: str) -> str:
    match = re.search(r"(\d+)pct", metric)
    return f"{match.group(1)}%" if match else metric


def results_table(metrics: pd.DataFrame) -> str:
    """Headline table: one row per model, test metrics with 95% bootstrap intervals."""
    r_at_p, p_at_r = _metric_names(metrics)
    models = metrics[metrics["role"].isin(["served", "candidate"])]
    order = [*models[models["role"] == "served"]["model"].unique()]
    order += [m for m in models["model"].unique() if m not in order]
    header = (
        f"| Model | Test PR-AUC (95% CI) | Recall @ {_pct_label(r_at_p)} precision "
        f"| Precision @ {_pct_label(p_at_r)} recall | ROC-AUC | Brier | Validation PR-AUC |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for m in order:
        served = (models[models["model"] == m]["role"] == "served").any()
        name = f"**{m}** (served)" if served else m
        lines.append(
            f"| {name} | {_ci(metrics, m, 'pr_auc')} | {_ci(metrics, m, r_at_p)} | "
            f"{_ci(metrics, m, p_at_r)} | {_ci(metrics, m, 'roc_auc')} | "
            f"{_ci(metrics, m, 'brier', 4)} | {_num(_value(metrics, m, 'pr_auc', 'validation'))} |"
        )
    return header + "\n".join(lines) + "\n"


def served_model(metrics: pd.DataFrame) -> str:
    return str(metrics[metrics["role"] == "served"]["model"].iloc[0])


def decision_table(metrics: pd.DataFrame) -> str:
    m = served_model(metrics)
    threshold = _value(metrics, m, "threshold", "validation")
    tp, fp, fn, tn = (int(_value(metrics, m, k)) for k in ("tp", "fp", "fn", "tn"))
    nothing = _value(metrics, "always_legit", "total_cost")
    everything = _value(metrics, "flag_everything", "total_cost")
    cost = _value(metrics, m, "total_cost")
    return (
        f"Decision threshold **{threshold:.4g}** (chosen on validation to minimise cost).\n\n"
        "| | Flagged | Allowed |\n|---|---|---|\n"
        f"| Fraud | {tp:,} (caught) | {fn:,} (missed) |\n"
        f"| Legitimate | {fp:,} (false alarms) | {tn:,} |\n\n"
        "| Policy on the test slice | Total cost | Saving by the model (95% CI) |\n"
        "|---|---|---|\n"
        f"| Flag nothing | {_money(nothing)} | "
        f"{_money(nothing - cost)}; {_pct_ci(metrics, m, 'savings_vs_flag_nothing_pct')} |\n"
        f"| Flag everything | {_money(everything)} | "
        f"{_money(everything - cost)}; {_pct_ci(metrics, m, 'savings_vs_flag_everything_pct')} |\n"
        f"| **{m}** at the chosen threshold | **{_money(cost)}** | |\n"
    )


def significance_text(comparisons: pd.DataFrame) -> str:
    if comparisons.empty:
        return ""
    lines = []
    for _, r in comparisons.iterrows():
        verdict = (
            "**significantly different**" if r["significant"] else "not significantly different"
        )
        lines.append(
            f"- vs `{r['model_b']}`: PR-AUC difference {r['difference']:+.3f} "
            f"(95% CI {r['ci_low']:+.3f} to {r['ci_high']:+.3f}), {verdict}."
        )
    served = comparisons.iloc[0]["model_a"]
    text = (
        f"Paired bootstrap of the PR-AUC difference, `{served}` minus each other model, on "
        "the same resampled test rows:\n\n" + "\n".join(lines) + "\n"
    )
    better = comparisons[comparisons["significant"] & (comparisons["ci_high"] < 0)]
    if not better.empty:
        names = ", ".join(f"`{n}`" for n in better["model_b"])
        text += (
            f"\n**The model chosen on validation is not the best on test:** {names} scored "
            "significantly higher on the test slice. The served model is kept anyway: it was "
            "selected by validation PR-AUC before the test slice was read, and switching now "
            "would be choosing a model on test data, which would make the test numbers "
            "optimistic. With so few validation frauds, model selection is itself noisy; this "
            "is reported as a finding, not corrected.\n"
        )
    return text


def accuracy_text(metrics: pd.DataFrame) -> str:
    m = served_model(metrics)
    legit = _value(metrics, "always_legit", "accuracy")
    model_acc = _value(metrics, m, "accuracy")
    tp, fn = int(_value(metrics, m, "tp")), int(_value(metrics, m, "fn"))
    return (
        f'Why not accuracy: predicting "legitimate" for every test transaction scores '
        f"**{100 * legit:.2f}% accuracy** and catches none of the {tp + fn} frauds. The served "
        f"model scores {100 * model_acc:.2f}% and catches {tp}. Accuracy separates them by "
        f"{100 * (model_acc - legit):.2f} percentage points, so it is reported only to make "
        "this point, never to compare models."
    )


def roc_text(metrics: pd.DataFrame) -> str:
    """ROC-AUC caveat, illustrated with the model whose ROC-AUC most flatters its PR-AUC."""
    test = metrics[(metrics["split"] == "test") & metrics["role"].isin(["served", "candidate"])]
    pivot = test.pivot_table(index="model", columns="metric", values="value")
    gap = str((pivot["roc_auc"] - pivot["pr_auc"]).idxmax())
    return (
        "ROC-AUC is reported for completeness only. With so few frauds, the false-positive "
        "rate's denominator is huge, so ROC-AUC stays high even for a model that is of little "
        f"use: `{gap}` reaches ROC-AUC {pivot.loc[gap, 'roc_auc']:.3f} but PR-AUC only "
        f"{pivot.loc[gap, 'pr_auc']:.3f}."
    )


def importance_text(importance: pd.DataFrame, k: int = 5) -> str:
    head = importance.head(k)
    items = ", ".join(
        f"`{r.feature}` ({r.importance_mean:.3f})" for r in head.itertuples(index=False)
    )
    return f"Top {k} by permutation importance (drop in validation PR-AUC): {items}."


def latency_text(reports: Path) -> str:
    path = reports / "load_test.json"
    if not path.exists():
        return "_No load test recorded yet. Run `python scripts/load_test.py`._\n"
    lt = json.loads(path.read_text(encoding="utf-8"))
    return (
        f"{lt['requests']:,} sequential `POST /score` requests against {lt['target']} "
        f"({lt['environment']}), measured client-side end to end:\n\n"
        "| p50 | p95 | p99 | max | errors |\n|---|---|---|---|---|\n"
        f"| {lt['p50_ms']:.1f} ms | **{lt['p95_ms']:.1f} ms** | {lt['p99_ms']:.1f} ms | "
        f"{lt['max_ms']:.1f} ms | {lt['errors']} |\n\n"
        f"Budget (NFR-2): p95 under {lt['budget_ms']:.0f} ms — "
        f"**{'met' if lt['p95_ms'] < lt['budget_ms'] else 'NOT met'}**. "
        f"Measured {lt['measured_at']}.\n"
    )


def example_text(reports: Path) -> str:
    path = reports / "example_response.json"
    if not path.exists():
        return "_No example response recorded yet. Run `python scripts/load_test.py`._\n"
    body = json.loads(path.read_text(encoding="utf-8"))
    return (
        "Response captured from the running container by `scripts/load_test.py`:\n\n"
        f"```json\n{json.dumps(body, indent=2)}\n```\n"
    )


def readme_results(reports: Path) -> str:
    metrics = pd.read_csv(reports / "metrics.csv")
    comparisons = pd.read_csv(reports / "comparisons.csv")
    importance = pd.read_csv(reports / "feature_importance.csv")
    return (
        "_Generated by `python -m fraudlens.report` from `reports/metrics.csv`. "
        "Do not edit by hand._\n\n"
        "Test-slice results (the final 20% of transactions by time, used once). Intervals are "
        f"95% percentile bootstrap intervals over {_bootstrap_samples(reports):,} resamples "
        "of test rows.\n\n"
        + results_table(metrics)
        + "\n"
        + significance_text(comparisons)
        + "\n"
        + decision_table(metrics)
        + "\n"
        + accuracy_text(metrics)
        + "\n\n"
        + roc_text(metrics)
        + "\n\n"
        + importance_text(importance)
        + "\n\n"
        "![Precision-recall curves](reports/pr_curves.png)\n"
        "![Cost by threshold](reports/cost_curve.png)\n"
    )


def replace_section(text: str, markers: tuple[str, str], body: str) -> str:
    start, end = markers
    if start not in text or end not in text:
        raise ValueError(f"README is missing the {start} ... {end} markers")
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    return f"{head}{start}\n{body.rstrip()}\n{end}{tail}"


def update_readme(readme: Path, reports: Path) -> None:
    text = readme.read_text(encoding="utf-8")
    if (reports / "metrics.csv").exists():
        text = replace_section(text, RESULTS, readme_results(reports))
    text = replace_section(text, LATENCY, latency_text(reports))
    if EXAMPLE[0] in text:
        text = replace_section(text, EXAMPLE, example_text(reports))
    readme.write_text(text, encoding="utf-8", newline="\n")


def _bootstrap_samples(reports: Path) -> int:
    """Read from ``reports/run.json``, which evaluate writes alongside metrics.csv."""
    return int(json.loads((reports / "run.json").read_text(encoding="utf-8"))["bootstrap_samples"])


@dataclass
class ReportContext:
    cfg: Config
    summary: dict[str, Any]
    metrics: pd.DataFrame
    comparisons: pd.DataFrame
    importance: pd.DataFrame
    costs: dict[str, Any]
    served_label: str
    evaluate_seconds: float


def _search_table(summary: dict[str, Any]) -> str:
    lines = [
        "| Model | Settings tried | Best validation PR-AUC | Best parameters | Search time |",
        "|---|---|---|---|---|",
    ]
    for name, m in summary["models"].items():
        params = ", ".join(f"{k}={v}" for k, v in sorted(m["params"].items()))
        lines.append(
            f"| {name} | {len(m['trials'])} | {m['validation_pr_auc']:.4f} | {params} | "
            f"{m['search_seconds']:.0f} s |"
        )
    return "\n".join(lines) + "\n"


def _imbalance_table(summary: dict[str, Any]) -> str:
    rows = summary["imbalance"]
    if not rows:
        return "_No imbalance comparison configured._\n"
    frame = pd.DataFrame(rows).pivot(index="model", columns="strategy", values="validation_pr_auc")
    cols = list(frame.columns)
    lines = ["| Model | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for model, r in frame.iterrows():
        best = r.idxmax()
        cells = [f"**{r[c]:.4f}**" if c == best else f"{r[c]:.4f}" for c in cols]
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _full_metrics_table(metrics: pd.DataFrame) -> str:
    test = metrics[(metrics["split"] == "test") & metrics["role"].isin(["served", "candidate"])]
    lines = ["| Model | Metric | Value | 95% CI |", "|---|---|---|---|"]
    for _, r in test.iterrows():
        low, high = r["ci_low"], r["ci_high"]
        ci = "" if pd.isna(low) else f"{float(low):.4g} – {float(high):.4g}"
        value = (
            f"{int(r['value']):,}"
            if r["metric"] in {"tp", "fp", "fn", "tn"}
            else f"{float(r['value']):.4g}"
        )
        lines.append(f"| {r['model']} | {r['metric']} | {value} | {ci} |")
    return "\n".join(lines) + "\n"


def _split_table(sp: dict[str, Any], costs: dict[str, Any]) -> str:
    rows = [
        ("Train", sp["train_rows"], sp["train_frauds"]),
        ("Validation", sp["validation_rows"], sp["validation_frauds"]),
        ("Test", costs["test_rows"], costs["test_frauds"]),
    ]
    lines = ["| Slice | Rows | Frauds | Fraud rate |", "|---|---|---|---|"]
    lines += [f"| {name} | {n:,} | {f} | {f / n:.4%} |" for name, n, f in rows]
    return "\n".join(lines) + "\n"


def write_report(ctx: ReportContext, path: Path) -> None:
    s = ctx.summary
    sp = s["split"]
    sel = s["selected"]
    c = ctx.costs
    cfg = ctx.cfg
    missed = (
        "the transaction amount"
        if cfg.costs.missed_fraud == "amount"
        else _money(float(cfg.costs.missed_fraud))
    )
    text = f"""# FraudLens evaluation report

_Generated by `python -m fraudlens.evaluate`. Every number below comes from the files in
`reports/` and the training artifacts of version `{s["version"]}`._

- Data SHA-256: `{s["data_sha256"]}`
- Code version: `{s["code_version"]}`
- Seed: {s["seed"]}
- Training time: {s["train_seconds"]:.0f} s; evaluation time: {ctx.evaluate_seconds:.0f} s

## 1. Split (EV-1)

Transactions are sorted by `Time` and cut 60/20/20, with each cut moved forward to the next
distinct timestamp so no second straddles two slices.

{_split_table(sp, c)}
The test slice was not read until every choice below had been made on validation data.

## 2. Model search on validation (FR-4 to FR-8)

Each setting is trained on the training slice and scored by validation PR-AUC.

{_search_table(s)}
## 3. Imbalance strategies (EV-5)

Validation PR-AUC for each supervised model's best parameters under each strategy.
Resampling happens inside the training pipeline only; validation and test rows are never
resampled. Undersampling and SMOTE target a minority/majority ratio of
{cfg.imbalance.undersample_ratio} and {cfg.imbalance.smote_ratio} respectively.

{_imbalance_table(s)}
## 4. Selection and calibration (FR-9)

Selected: **{sel["model"]}** ({sel["selection_rule"]}), calibrated with
**{sel["calibration"]}** scaling fitted on the validation slice. The calibration map is
monotonic, so it leaves the ranking (and PR-AUC) unchanged and only makes the scores usable
as probabilities.

## 5. Test results (EV-2 to EV-4)

{results_table(ctx.metrics)}
{significance_text(ctx.comparisons)}
{roc_text(ctx.metrics)}

{accuracy_text(ctx.metrics)}

![PR curves](pr_curves.png)

![Calibration](calibration.png)

## 6. Cost-based threshold (EV-7)

Cost model (`config.yaml`): a missed fraud costs {missed}; a false alarm costs
{_money(cfg.costs.false_alarm)} (one review); a caught fraud costs
{_money(cfg.costs.caught_fraud)} (it is reviewed too). The threshold that minimised total
cost on validation was applied unchanged to the test slice.

{decision_table(ctx.metrics)}
On validation, the chosen threshold cost {_money(c["validation_outcome"]["total_cost"])} with
{c["validation_outcome"]["tp"]} frauds caught, {c["validation_outcome"]["fn"]} missed and
{c["validation_outcome"]["fp"]} false alarms. Test frauds totalled
{_money(c["test_fraud_amount"])}; the money figures describe this dataset only.

![Cost curve](cost_curve.png)

## 7. Explainability (FR-13, FR-14)

{importance_text(ctx.importance, 10)}

`V1`–`V28` are anonymised PCA components, so these rankings say *which component* carries
signal, not *what it means* in business terms. `Time` enters the model only as the hour of
day and `Amount` as log1p(Amount). The API returns per-transaction contributions computed by
replacing each input with its training median and re-scoring.

![Permutation importance](feature_importance.png)

## 8. All test metrics

{_full_metrics_table(ctx.metrics)}
## 9. Limitations

- Two days of 2013 transactions from one European issuer. Nothing here is evidence of
  performance on another portfolio, another period, or in production.
- The test slice holds only {c["test_frauds"]} frauds, so every interval is wide; small
  differences between models are noise unless the paired interval excludes zero.
- PCA features limit explanations to component names.
- Labels are taken as given and arrive instantly; real fraud labels arrive weeks later and
  are incomplete (unreported fraud), and fraud patterns drift.
- One global threshold is used, as specified. Because a missed fraud costs its amount, an
  amount-aware rule (flag when probability × amount exceeds the review cost) would likely do
  better; it is left as a next step.
"""
    path.write_text(text, encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Refresh README results from reports/.")
    parser.add_argument("--config", default="config.yaml", type=Path)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    update_readme(cfg.readme_path, cfg.reports_dir)


if __name__ == "__main__":
    main()
