import clsx from "clsx";
import { BarChart3, Table2 } from "lucide-react";
import { animate, useMotionValue, useReducedMotion } from "motion/react";
import { useEffect, useState, type ReactNode } from "react";
import { Card, CardHeader, Segmented } from "../ui";

/**
 * Every chart has a table twin (the accessible, colour-free reading of the same numbers).
 * The toggle lives in the card header; the chart never gates a value behind a tooltip.
 */
export function ChartCard({
  eyebrow,
  title,
  description,
  chart,
  table,
  legend,
  actions,
  className,
}: {
  eyebrow?: string;
  title: ReactNode;
  description?: ReactNode;
  chart: ReactNode;
  table: ReactNode;
  legend?: ReactNode;
  actions?: ReactNode;
  className?: string;
}) {
  const [view, setView] = useState<"chart" | "table">("chart");
  return (
    <Card className={className}>
      <CardHeader
        eyebrow={eyebrow}
        title={title}
        description={description}
        actions={
          <>
            {actions}
            <Segmented
              label="Chart or table view"
              value={view}
              onChange={setView}
              options={[
                { value: "chart", label: <BarChart3 size={13} aria-label="Chart" /> },
                { value: "table", label: <Table2 size={13} aria-label="Table" /> },
              ]}
            />
          </>
        }
      />
      {view === "chart" ? (
        <>
          {legend && <div className="mb-3">{legend}</div>}
          {chart}
        </>
      ) : (
        <div className="max-h-[420px] overflow-auto rounded-lg ring-1 ring-inset ring-line">
          {table}
        </div>
      )}
    </Card>
  );
}

export function DataTable({
  columns,
  rows,
  caption,
}: {
  columns: { key: string; label: ReactNode; align?: "left" | "right" }[];
  rows: Record<string, ReactNode>[];
  caption?: string;
}) {
  return (
    <table className="w-full border-collapse text-[12px]">
      {caption && <caption className="sr-only">{caption}</caption>}
      <thead className="sticky top-0 bg-surface-2">
        <tr>
          {columns.map((c) => (
            <th
              key={c.key}
              scope="col"
              className={clsx(
                "border-b border-line px-3 py-2 font-medium text-ink-3",
                c.align === "right" ? "text-right" : "text-left",
              )}
            >
              {c.label}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((r, i) => (
          <tr key={i} className="border-b border-line last:border-0 hover:bg-surface-2">
            {columns.map((c) => (
              <td
                key={c.key}
                className={clsx(
                  "px-3 py-1.5 text-ink-2",
                  c.align === "right" ? "tnum text-right" : "text-left",
                )}
              >
                {r[c.key]}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/** Legend chips that also toggle series. Hidden series keep their colour slot. */
export function LegendToggle({
  items,
  hidden,
  onToggle,
}: {
  items: { id: string; label: ReactNode; color: string; note?: ReactNode }[];
  hidden: Set<string>;
  onToggle: (id: string) => void;
}) {
  return (
    <ul className="flex flex-wrap gap-1.5" aria-label="Series">
      {items.map((it) => {
        const off = hidden.has(it.id);
        return (
          <li key={it.id}>
            <button
              type="button"
              aria-pressed={!off}
              onClick={() => onToggle(it.id)}
              className={clsx(
                "inline-flex items-center gap-2 rounded-full px-2.5 py-1 text-[12px] ring-1 ring-inset transition-all",
                off
                  ? "text-ink-3 ring-line opacity-60"
                  : "bg-surface-2 text-ink ring-line-strong",
              )}
            >
              <span
                className="h-2 w-3.5 rounded-full"
                style={{ background: off ? "var(--axis)" : it.color }}
                aria-hidden
              />
              {it.label}
              {it.note && <span className="tnum text-ink-3">{it.note}</span>}
            </button>
          </li>
        );
      })}
    </ul>
  );
}

/** Smoothly animated number (skips animation when reduced motion is requested). */
export function AnimatedNumber({
  value,
  format,
  className,
}: {
  value: number;
  format: (v: number) => string;
  className?: string;
}) {
  const reduce = useReducedMotion();
  const mv = useMotionValue(value);
  const [shown, setShown] = useState(value);
  useEffect(() => {
    if (reduce) return;
    const controls = animate(mv, value, {
      duration: 0.45,
      ease: [0.22, 1, 0.36, 1],
      onUpdate: setShown,
    });
    return () => controls.stop();
  }, [value, mv, reduce]);
  return <span className={className}>{format(reduce ? value : shown)}</span>;
}

/** Shared tooltip chrome for Recharts. */
export function TooltipBox({ title, rows }: { title: ReactNode; rows: { label: ReactNode; value: ReactNode; color?: string }[] }) {
  return (
    <div className="min-w-44 rounded-xl border border-line-strong bg-surface/95 px-3 py-2.5 text-[12px] shadow-xl backdrop-blur">
      <p className="mb-1.5 font-medium text-ink">{title}</p>
      <ul className="space-y-1">
        {rows.map((r, i) => (
          <li key={i} className="flex items-center gap-2 text-ink-2">
            {r.color && <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: r.color }} />}
            <span className="truncate">{r.label}</span>
            <span className="tnum ml-auto pl-3 font-medium text-ink">{r.value}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

