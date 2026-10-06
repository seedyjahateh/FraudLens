import type { Sweep } from "./schemas";

/** Review costs the explorer lets the viewer change. A missed fraud always costs its amount. */
export interface ReviewCosts {
  falseAlarm: number;
  caughtFraud: number;
}

export interface PointMetrics {
  index: number;
  threshold: number;
  tp: number;
  fp: number;
  fn: number;
  tn: number;
  flagged: number;
  precision: number;
  recall: number;
  falsePositiveRate: number;
  missedCost: number;
  reviewCost: number;
  totalCost: number;
  flagNothingCost: number;
  flagEverythingCost: number;
  savedVsNothing: number;
  savedVsNothingPct: number;
  savedVsEverything: number;
  savedVsEverythingPct: number;
}

/** Same formula as fraudlens.costs.Sweep.totals. */
export function totalCost(sweep: Sweep, i: number, costs: ReviewCosts): number {
  return (
    (sweep.missed_cost[i] ?? 0) +
    (sweep.fp[i] ?? 0) * costs.falseAlarm +
    (sweep.tp[i] ?? 0) * costs.caughtFraud
  );
}

export function baselines(sweep: Sweep, costs: ReviewCosts): { nothing: number; everything: number } {
  const legit = sweep.rows - sweep.frauds;
  return {
    nothing: sweep.missed_cost[0] ?? 0,
    everything: legit * costs.falseAlarm + sweep.frauds * costs.caughtFraud,
  };
}

export function pointMetrics(sweep: Sweep, i: number, costs: ReviewCosts): PointMetrics {
  const at = <K extends "tp" | "fp" | "fn" | "tn">(k: K) => sweep[k][i] ?? 0;
  const tp = at("tp");
  const fp = at("fp");
  const fn = at("fn");
  const tn = at("tn");
  const flagged = tp + fp;
  const total = totalCost(sweep, i, costs);
  const base = baselines(sweep, costs);
  const missedCost = sweep.missed_cost[i] ?? 0;
  return {
    index: i,
    threshold: sweep.threshold[i] ?? Number.NaN,
    tp,
    fp,
    fn,
    tn,
    flagged,
    precision: flagged ? tp / flagged : 0,
    recall: tp + fn ? tp / (tp + fn) : 0,
    falsePositiveRate: fp + tn ? fp / (fp + tn) : 0,
    missedCost,
    reviewCost: total - missedCost,
    totalCost: total,
    flagNothingCost: base.nothing,
    flagEverythingCost: base.everything,
    savedVsNothing: base.nothing - total,
    savedVsNothingPct: base.nothing > 0 ? (100 * (base.nothing - total)) / base.nothing : 0,
    savedVsEverything: base.everything - total,
    savedVsEverythingPct:
      base.everything > 0 ? (100 * (base.everything - total)) / base.everything : 0,
  };
}

/** Index of the cheapest point; ties go to the higher threshold (fewer alerts). */
export function minCostIndex(sweep: Sweep, costs: ReviewCosts): number {
  let best = 0;
  let bestCost = Number.POSITIVE_INFINITY;
  for (let i = 0; i < sweep.threshold.length; i++) {
    const c = totalCost(sweep, i, costs);
    if (c < bestCost) {
      bestCost = c;
      best = i;
    }
  }
  return best;
}

/** The sweep point whose flags equal "score >= threshold" (thresholds decrease). */
export function indexForThreshold(sweep: Sweep, threshold: number): number {
  let found = 0;
  for (let i = 0; i < sweep.threshold.length; i++) {
    if ((sweep.threshold[i] ?? -1) >= threshold) found = i;
    else break;
  }
  return found;
}

/** Chart-friendly rows: x on a log scale needs strictly positive values. */
export function curveRows(sweep: Sweep, costs: ReviewCosts, floor = 1e-6) {
  return sweep.threshold.map((t, i) => ({
    index: i,
    threshold: Math.min(Math.max(t, floor), 1),
    cost: totalCost(sweep, i, costs),
    recall: sweep.frauds ? (sweep.tp[i] ?? 0) / sweep.frauds : 0,
    flagged: (sweep.tp[i] ?? 0) + (sweep.fp[i] ?? 0),
  }));
}
