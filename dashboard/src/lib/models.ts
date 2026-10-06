import type { Bundle, MetricRow } from "./schemas";

/*
 * Colour follows the entity, never its rank: each model id owns one categorical slot
 * (validated order), so hiding a series never repaints the others.
 */
const SLOT: Record<string, number> = {
  served: 1,
  logistic_regression: 2,
  random_forest: 3,
  hist_gradient_boosting: 4,
  isolation_forest: 5,
};

export function modelColor(id: string, servedLabel: string): string {
  const slot = id === servedLabel ? SLOT.served : (SLOT[id] ?? 5);
  return `var(--series-${slot})`;
}

/** Served model first, then candidates in a fixed order. */
export function modelOrder(bundle: Bundle): string[] {
  const fixed = ["logistic_regression", "random_forest", "hist_gradient_boosting", "isolation_forest"];
  const present = new Set(bundle.metrics.filter((r) => r.role !== "baseline").map((r) => r.model));
  return [bundle.meta.served_label, ...fixed.filter((m) => present.has(m))];
}

export function metric(
  bundle: Bundle,
  model: string,
  name: string,
  split: "test" | "validation" = "test",
): MetricRow | undefined {
  return bundle.metrics.find((r) => r.model === model && r.metric === name && r.split === split);
}

export function metricValue(
  bundle: Bundle,
  model: string,
  name: string,
  split: "test" | "validation" = "test",
): number {
  return metric(bundle, model, name, split)?.value ?? Number.NaN;
}

/** Metric ids whose names carry the configured targets, e.g. recall_at_90pct_precision. */
export function targetMetrics(bundle: Bundle): { recallAtP: string; precisionAtR: string } {
  const names = new Set(bundle.metrics.map((r) => r.metric));
  const recallAtP = [...names].find((m) => /^recall_at_\d+pct_precision$/.test(m)) ?? "";
  const precisionAtR = [...names].find((m) => /^precision_at_\d+pct_recall$/.test(m)) ?? "";
  return { recallAtP, precisionAtR };
}

export function targetLabel(metricName: string): string {
  const m = /(\d+)pct/.exec(metricName);
  return m ? `${m[1]}%` : "";
}
