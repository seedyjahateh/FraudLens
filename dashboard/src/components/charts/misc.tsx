import clsx from "clsx";
import { motion, useReducedMotion } from "motion/react";
import { useMemo } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { fmt, modelName } from "../../lib/format";
import { modelColor } from "../../lib/models";
import type { Bundle } from "../../lib/schemas";
import { AXIS_TICK } from "../../lib/nav";
import { AnimatedNumber, TooltipBox } from "./common";

/* ---------------------------------------------------------------- confusion matrix */

export function ConfusionMatrix({
  tp,
  fp,
  fn,
  tn,
}: {
  tp: number;
  fp: number;
  fn: number;
  tn: number;
}) {
  const cell = (
    value: number,
    label: string,
    tone: "good" | "critical" | "warning" | "neutral",
    hint: string,
  ) => (
    <div
      className={clsx(
        "rounded-xl p-4 ring-1 ring-inset",
        tone === "good" && "bg-good/8 ring-good/25",
        tone === "critical" && "bg-critical/8 ring-critical/25",
        tone === "warning" && "bg-warning/8 ring-warning/25",
        tone === "neutral" && "bg-surface-2 ring-line",
      )}
    >
      <p className="eyebrow">{label}</p>
      <AnimatedNumber
        value={value}
        format={fmt.int}
        className="mt-2 block text-[24px] font-semibold tracking-[-0.02em] text-ink"
      />
      <p className="mt-1 text-[11px] text-ink-3">{hint}</p>
    </div>
  );
  return (
    <div className="grid grid-cols-[auto_1fr_1fr] gap-2" role="table" aria-label="Confusion matrix">
      <span />
      <p className="eyebrow pb-1 text-center" role="columnheader">Flagged</p>
      <p className="eyebrow pb-1 text-center" role="columnheader">Allowed</p>
      <p className="eyebrow self-center pr-1 [writing-mode:vertical-rl] rotate-180" role="rowheader">Fraud</p>
      {cell(tp, "Caught", "good", "fraud stopped, one review each")}
      {cell(fn, "Missed", "critical", "fraud let through, costs its amount")}
      <p className="eyebrow self-center pr-1 [writing-mode:vertical-rl] rotate-180" role="rowheader">Legit</p>
      {cell(fp, "False alarms", "warning", "one wasted review each")}
      {cell(tn, "Allowed", "neutral", "no cost")}
    </div>
  );
}

/* ---------------------------------------------------------------- probability gauge */

export function ProbabilityGauge({
  probability,
  threshold,
}: {
  probability: number;
  threshold: number;
}) {
  const reduce = useReducedMotion();
  const r = 80;
  const c = Math.PI * r;
  const flagged = probability >= threshold;
  const angle = (v: number) => Math.PI * (1 - Math.min(Math.max(v, 0), 1));
  const tick = angle(threshold);
  return (
    <div className="relative mx-auto w-full max-w-[260px]">
      <svg viewBox="0 0 200 116" className="w-full" role="img" aria-label={`Fraud probability ${fmt.pct(probability, 2)}`}>
        <path
          d={`M 20 100 A ${r} ${r} 0 0 1 180 100`}
          fill="none"
          stroke="var(--surface-3)"
          strokeWidth="12"
          strokeLinecap="round"
        />
        <motion.path
          d={`M 20 100 A ${r} ${r} 0 0 1 180 100`}
          fill="none"
          stroke={flagged ? "var(--critical)" : "var(--series-1)"}
          strokeWidth="12"
          strokeLinecap="round"
          strokeDasharray={c}
          initial={{ strokeDashoffset: c }}
          animate={{ strokeDashoffset: c * (1 - Math.min(Math.max(probability, 0.004), 1)) }}
          transition={{ duration: reduce ? 0 : 0.7, ease: [0.22, 1, 0.36, 1] }}
        />
        <line
          x1={100 + (r - 14) * Math.cos(tick)}
          y1={100 - (r - 14) * Math.sin(tick)}
          x2={100 + (r + 10) * Math.cos(tick)}
          y2={100 - (r + 10) * Math.sin(tick)}
          stroke="var(--ink)"
          strokeWidth="2"
          strokeLinecap="round"
        />
      </svg>
      <div className="absolute inset-x-0 bottom-1 text-center">
        <AnimatedNumber
          value={probability * 100}
          format={(v) => `${v < 1 ? v.toFixed(2) : v.toFixed(1)}%`}
          className="text-[30px] font-semibold tracking-[-0.03em] text-ink"
        />
        <p className="text-[11px] text-ink-3">fraud probability · threshold {fmt.pct(threshold, 1)}</p>
      </div>
    </div>
  );
}

/* ---------------------------------------------------------------- contributions */

export function ContributionBars({
  items,
}: {
  items: { feature: string; contribution: number }[];
}) {
  const max = Math.max(1e-9, ...items.map((i) => Math.abs(i.contribution)));
  return (
    <ul className="space-y-2" aria-label="Top contributing features">
      {items.map((it, k) => {
        const width = `${(50 * Math.abs(it.contribution)) / max}%`;
        const up = it.contribution >= 0;
        return (
          <li key={it.feature} className="grid grid-cols-[64px_1fr_86px] items-center gap-3">
            <span className="font-mono text-[12px] text-ink">{it.feature}</span>
            <div className="relative h-6 rounded-md bg-surface-2">
              <span className="absolute inset-y-0 left-1/2 w-px bg-axis" aria-hidden />
              <motion.span
                className="absolute inset-y-1 rounded"
                style={{
                  background: up ? "var(--diverge-pos)" : "var(--diverge-neg)",
                  [up ? "left" : "right"]: "50%",
                }}
                initial={{ width: 0 }}
                animate={{ width }}
                transition={{ duration: 0.5, delay: k * 0.05, ease: [0.22, 1, 0.36, 1] }}
                aria-hidden
              />
            </div>
            <span className="tnum text-right text-[12px] text-ink-2">
              {up ? "+" : "−"}
              {(Math.abs(it.contribution) * 100).toFixed(Math.abs(it.contribution) < 0.001 ? 3 : 1)} pts
            </span>
          </li>
        );
      })}
      <li className="flex justify-between pt-1 text-[11px] text-ink-3">
        <span>← pushes towards legitimate</span>
        <span>pushes towards fraud →</span>
      </li>
    </ul>
  );
}

/* ---------------------------------------------------------------- sparkline */

export function Sparkline({ values, height = 56 }: { values: number[]; height?: number }) {
  const path = useMemo(() => {
    if (values.length < 2) return "";
    const max = Math.max(...values);
    const min = Math.min(...values);
    const span = max - min || 1;
    return values
      .map((v, i) => {
        const x = (i / (values.length - 1)) * 100;
        const y = 4 + (1 - (v - min) / span) * (height - 8);
        return `${i ? "L" : "M"}${x.toFixed(2)},${y.toFixed(2)}`;
      })
      .join(" ");
  }, [values, height]);
  if (!path) {
    return (
      <div className="flex items-center justify-center text-[12px] text-ink-3" style={{ height }}>
        Waiting for /score traffic…
      </div>
    );
  }
  return (
    <svg
      viewBox={`0 0 100 ${height}`}
      preserveAspectRatio="none"
      className="w-full"
      style={{ height }}
      role="img"
      aria-label="Recent /score latency"
    >
      <path d={`${path} L100,${height} L0,${height} Z`} fill="var(--accent-soft)" />
      <path d={path} fill="none" stroke="var(--series-1)" strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

/* ---------------------------------------------------------------- importance */

export function ImportanceBars({
  rows,
  top = 12,
}: {
  rows: Bundle["importance"];
  top?: number;
}) {
  const head = rows.slice(0, top);
  const max = Math.max(1e-9, ...head.map((r) => r.importance_mean + r.importance_std));
  return (
    <ul className="space-y-1.5" aria-label="Permutation importance">
      {head.map((r) => (
        <li key={r.feature} className="grid grid-cols-[64px_1fr_56px] items-center gap-3">
          <span className="font-mono text-[12px] text-ink-2">{r.feature}</span>
          <div className="relative h-5">
            <span
              className="absolute inset-y-1 left-0 rounded-r"
              style={{
                width: `${(100 * Math.max(r.importance_mean, 0)) / max}%`,
                background: "var(--series-1)",
              }}
              aria-hidden
            />
            <span
              className="absolute top-1/2 h-px -translate-y-1/2 bg-ink-3"
              style={{
                left: `${(100 * Math.max(r.importance_mean - r.importance_std, 0)) / max}%`,
                width: `${(100 * 2 * r.importance_std) / max}%`,
              }}
              aria-hidden
            />
          </div>
          <span className="tnum text-right text-[12px] text-ink-2">{r.importance_mean.toFixed(3)}</span>
        </li>
      ))}
    </ul>
  );
}

/* ---------------------------------------------------------------- imbalance heatmap */

const RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"];

export function ImbalanceHeatmap({ rows }: { rows: Bundle["imbalance"] }) {
  const models = [...new Set(rows.map((r) => r.model))];
  const strategies = [...new Set(rows.map((r) => r.strategy))];
  const values = rows.map((r) => r.validation_pr_auc);
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const shade = (v: number) => {
    const k = hi > lo ? Math.round(((v - lo) / (hi - lo)) * (RAMP.length - 1)) : RAMP.length - 1;
    return { bg: RAMP[k] ?? RAMP[0], ink: k >= 3 ? "#ffffff" : "#0b0b0b" };
  };
  const label: Record<string, string> = {
    class_weight: "Class weights",
    undersample: "Undersampling",
    smote: "SMOTE",
  };
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-separate border-spacing-1 text-[12px]">
        <caption className="sr-only">Validation PR-AUC by imbalance strategy</caption>
        <thead>
          <tr>
            <th />
            {strategies.map((s) => (
              <th key={s} scope="col" className="px-2 pb-1 text-center font-medium text-ink-3">
                {label[s] ?? s}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {models.map((m) => {
            const mine = rows.filter((r) => r.model === m);
            const best = Math.max(...mine.map((r) => r.validation_pr_auc));
            return (
              <tr key={m}>
                <th scope="row" className="pr-2 text-left font-medium text-ink-2">
                  {modelName(m)}
                </th>
                {strategies.map((s) => {
                  const v = mine.find((r) => r.strategy === s)?.validation_pr_auc;
                  if (v == null) return <td key={s} />;
                  const { bg, ink } = shade(v);
                  return (
                    <td
                      key={s}
                      className="tnum h-10 rounded-md text-center"
                      style={{ background: bg, color: ink }}
                    >
                      {v === best ? <strong>{v.toFixed(3)} ★</strong> : v.toFixed(3)}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
      <div className="mt-2 flex items-center gap-2 text-[11px] text-ink-3">
        <span className="tnum">{lo.toFixed(3)}</span>
        <span className="flex h-2 flex-1 overflow-hidden rounded-full">
          {RAMP.map((c) => (
            <span key={c} className="flex-1" style={{ background: c }} />
          ))}
        </span>
        <span className="tnum">{hi.toFixed(3)}</span>
        <span>· ★ best strategy per model</span>
      </div>
    </div>
  );
}

/* ---------------------------------------------------------------- calibration */

export function CalibrationChart({ bundle }: { bundle: Bundle }) {
  const served = bundle.meta.served_label;
  const names = Object.keys(bundle.calibration);
  const lines = names.map((m) => {
    const c = bundle.calibration[m];
    return {
      id: m,
      points: (c?.predicted ?? []).map((p, i) => ({ predicted: p, [m]: c?.observed[i] ?? null })),
    };
  });
  const histogram = (bundle.calibration[served]?.histogram ?? []).map((count, i, arr) => ({
    bin: (i + 0.5) / arr.length,
    count: Math.max(count, 0.8),
    raw: count,
  }));
  return (
    <div>
      <ul className="mb-3 flex flex-wrap gap-x-4 gap-y-1.5 text-[12px] text-ink-2" aria-label="Series">
        {names.map((m) => (
          <li key={m} className="flex items-center gap-2">
            <span className="h-2 w-3.5 rounded-full" style={{ background: modelColor(m, served) }} aria-hidden />
            {modelName(m)}
          </li>
        ))}
        <li className="flex items-center gap-2 text-ink-3">
          <span className="w-3.5 border-t border-dotted border-ink-3" aria-hidden /> perfectly calibrated
        </li>
      </ul>
      <div className="h-[240px]" role="img" aria-label="Calibration: predicted probability against observed fraud rate">
        <ResponsiveContainer>
          <LineChart margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
            <CartesianGrid vertical={false} />
            <XAxis
              dataKey="predicted"
              type="number"
              domain={[0, 1]}
              ticks={[0, 0.25, 0.5, 0.75, 1]}
              tickFormatter={(v: number) => fmt.pct(v, 0)}
              tick={AXIS_TICK}
              allowDuplicatedCategory={false}
            />
            <YAxis domain={[0, 1]} tickFormatter={(v: number) => fmt.pct(v, 0)} tick={AXIS_TICK} width={44} />
            <ReferenceLine
              segment={[
                { x: 0, y: 0 },
                { x: 1, y: 1 },
              ]}
              stroke="var(--ink-3)"
              strokeDasharray="2 4"
            />
            <Tooltip
              content={({ active, payload }) =>
                active && payload?.length ? (
                  <TooltipBox
                    title={`Predicted ${fmt.pct(Number(payload[0]?.payload?.predicted), 1)}`}
                    rows={payload.map((p) => ({
                      label: modelName(String(p.dataKey)),
                      value: `observed ${fmt.pct(Number(p.value), 1)}`,
                      color: modelColor(String(p.dataKey), served),
                    }))}
                  />
                ) : null
              }
            />
            {lines.map((l) => (
              <Line
                key={l.id}
                data={l.points}
                dataKey={l.id}
                stroke={modelColor(l.id, served)}
                strokeWidth={l.id === served ? 2.5 : 1.5}
                strokeOpacity={l.id === served ? 1 : 0.7}
                dot={{ r: 3, strokeWidth: 0, fill: modelColor(l.id, served) }}
                isAnimationActive={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
      <p className="eyebrow mb-1 mt-4">Scores of the served model (log count)</p>
      <div className="h-[90px]" role="img" aria-label="Histogram of served model scores">
        <ResponsiveContainer>
          <BarChart data={histogram} margin={{ top: 0, right: 12, bottom: 0, left: 0 }} barCategoryGap={2}>
            <XAxis dataKey="bin" hide />
            <YAxis scale="log" domain={[0.8, "auto"]} hide />
            <Tooltip
              cursor={{ fill: "var(--surface-3)" }}
              content={({ active, payload }) => {
                const p = payload?.[0]?.payload as (typeof histogram)[number] | undefined;
                return active && p ? (
                  <TooltipBox
                    title={`Score ${fmt.pct(p.bin - 0.025, 0)}–${fmt.pct(p.bin + 0.025, 0)}`}
                    rows={[{ label: "Transactions", value: fmt.int(p.raw) }]}
                  />
                ) : null;
              }}
            />
            <Bar dataKey="count" fill="var(--series-1)" radius={[3, 3, 0, 0]} isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
