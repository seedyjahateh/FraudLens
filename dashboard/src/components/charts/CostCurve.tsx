import { useMemo, useRef } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceDot,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { fmt } from "../../lib/format";
import type { Sweep } from "../../lib/schemas";
import { curveRows, type ReviewCosts } from "../../lib/threshold";
import { AXIS_TICK } from "../../lib/nav";
import { TooltipBox } from "./common";

function niceCeil(v: number): number {
  const step = 10 ** Math.floor(Math.log10(Math.max(v, 1)));
  for (const m of [1, 2, 2.5, 5, 10]) if (m * step >= v) return m * step;
  return 10 * step;
}

/**
 * Total cost against the decision threshold (log scale). Click or drag on the plot to move
 * the threshold; the chosen (validation) threshold is marked and labelled.
 */
export function CostCurve({
  sweep,
  costs,
  index,
  chosenIndex,
  onSelect,
}: {
  sweep: Sweep;
  costs: ReviewCosts;
  index: number;
  chosenIndex: number;
  onSelect: (index: number) => void;
}) {
  const rows = useMemo(() => curveRows(sweep, costs), [sweep, costs]);
  const dragging = useRef(false);
  const current = rows[index];
  const chosen = rows[chosenIndex];
  // Flagging almost everything costs orders of magnitude more; cap the view just above
  // the flag-nothing cost so the region around the optimum stays readable.
  const yMax = niceCeil((rows[0]?.cost ?? 1) * 1.2);

  const pick = (state: { activeTooltipIndex?: number | string | null } | null) => {
    const i = Number(state?.activeTooltipIndex);
    if (Number.isInteger(i) && i >= 0 && i < rows.length) onSelect(i);
  };

  return (
    <div
      className="h-[300px] w-full cursor-crosshair select-none"
      role="img"
      aria-label="Total cost by decision threshold"
    >
      <ResponsiveContainer>
        <ComposedChart
          data={rows}
          margin={{ top: 16, right: 16, bottom: 18, left: 8 }}
          onMouseDown={(s) => {
            dragging.current = true;
            pick(s);
          }}
          onMouseMove={(s) => dragging.current && pick(s)}
          onMouseUp={() => {
            dragging.current = false;
          }}
          onMouseLeave={() => {
            dragging.current = false;
          }}
        >
          <defs>
            <linearGradient id="costFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="var(--series-1)" stopOpacity={0.18} />
              <stop offset="100%" stopColor="var(--series-1)" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid vertical={false} />
          <XAxis
            dataKey="threshold"
            type="number"
            scale="log"
            domain={[1e-6, 1]}
            reversed
            allowDataOverflow
            ticks={[1, 0.1, 0.01, 0.001, 1e-4, 1e-5, 1e-6]}
            tickFormatter={(v: number) => (v >= 0.01 ? String(v) : v.toExponential(0))}
            tick={AXIS_TICK}
            label={{ value: "Threshold (flag if probability ≥ threshold) → lower = more alerts", position: "insideBottom", offset: -12 }}
          />
          <YAxis
            domain={[0, yMax]}
            allowDataOverflow
            tickFormatter={(v: number) => fmt.money(v)}
            tick={AXIS_TICK}
            width={64}
          />
          <Tooltip
            cursor={{ stroke: "var(--axis)", strokeWidth: 1 }}
            content={({ active, payload }) => {
              const p = payload?.[0]?.payload as (typeof rows)[number] | undefined;
              return active && p ? (
                <TooltipBox
                  title={`Threshold ${fmt.threshold(sweep.threshold[p.index] ?? p.threshold)}`}
                  rows={[
                    { label: "Total cost", value: fmt.money(p.cost) },
                    { label: "Alerts", value: fmt.int(p.flagged) },
                    { label: "Fraud caught", value: fmt.pct(p.recall, 0) },
                  ]}
                />
              ) : null;
            }}
          />
          <Area
            dataKey="cost"
            type="stepAfter"
            stroke="none"
            fill="url(#costFill)"
            isAnimationActive={false}
          />
          <Line
            dataKey="cost"
            type="stepAfter"
            stroke="var(--series-1)"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, strokeWidth: 2, stroke: "var(--surface)" }}
            isAnimationActive={false}
          />
          {chosen && (
            <ReferenceLine
              x={chosen.threshold}
              stroke="var(--ink-3)"
              strokeWidth={1}
              label={{
                value: "chosen on validation",
                position: "insideTopLeft",
                fill: "var(--ink-3)",
                fontSize: 11,
              }}
            />
          )}
          {current && (
            <>
              <ReferenceLine x={current.threshold} stroke="var(--series-1)" strokeWidth={1.5} />
              <ReferenceDot
                x={current.threshold}
                y={Math.min(current.cost, yMax)}
                r={6}
                fill="var(--series-1)"
                stroke="var(--surface)"
                strokeWidth={2.5}
              />
            </>
          )}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
