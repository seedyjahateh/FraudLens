const intFmt = new Intl.NumberFormat("en-US");
const moneyFmt = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});
const moneyCents = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

export const fmt = {
  int: (v: number) => intFmt.format(Math.round(v)),
  num: (v: number, digits = 3) => (Number.isFinite(v) ? v.toFixed(digits) : "–"),
  pct: (v: number, digits = 1) => (Number.isFinite(v) ? `${(v * 100).toFixed(digits)}%` : "–"),
  /** Values already in percent units (e.g. savings_pct). */
  pctUnits: (v: number, digits = 1) => (Number.isFinite(v) ? `${v.toFixed(digits)}%` : "–"),
  money: (v: number) => (Number.isFinite(v) ? moneyFmt.format(v) : "–"),
  moneyCents: (v: number) => (Number.isFinite(v) ? moneyCents.format(v) : "–"),
  signed: (v: number, digits = 3) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(digits)}`,
  ms: (v: number | null | undefined) =>
    v == null || !Number.isFinite(v) ? "–" : v < 10 ? `${v.toFixed(1)} ms` : `${Math.round(v)} ms`,
  /** Thresholds span 1e-6..1, so show significant digits rather than fixed decimals. */
  threshold: (v: number) => {
    if (v > 1) return "> max score";
    if (v === 0) return "0";
    if (v < 0.001) return v.toExponential(1);
    return v.toPrecision(3);
  },
  ci: (low: number | null, high: number | null, digits = 3) =>
    low == null || high == null ? "" : `${low.toFixed(digits)}–${high.toFixed(digits)}`,
  duration: (seconds: number) => {
    if (seconds < 60) return `${Math.round(seconds)} s`;
    const m = Math.floor(seconds / 60);
    if (m < 60) return `${m} min ${Math.round(seconds % 60)} s`;
    const h = Math.floor(m / 60);
    return `${h} h ${m % 60} min`;
  },
  date: (iso: string) => {
    const d = new Date(iso);
    return Number.isNaN(d.getTime())
      ? iso
      : d.toLocaleString("en-US", { dateStyle: "medium", timeStyle: "short" });
  },
  shortHash: (v: string) => (v.length > 12 ? `${v.slice(0, 8)}…${v.slice(-4)}` : v),
};

/** Human names for model ids. The served one keeps its "(calibrated)" suffix. */
export function modelName(id: string): string {
  const base: Record<string, string> = {
    logistic_regression: "Logistic regression",
    random_forest: "Random forest",
    hist_gradient_boosting: "Gradient boosting",
    isolation_forest: "Isolation forest",
    always_legit: "Always “legitimate”",
    flag_everything: "Flag everything",
  };
  const match = /^(.*) \(calibrated\)$/.exec(id);
  if (match?.[1]) return `${base[match[1]] ?? match[1]} · calibrated`;
  return base[id] ?? id;
}
