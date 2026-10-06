import { useMemo, useState } from "react";
import {
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
import { metricValue, modelColor, modelOrder } from "../../lib/models";
import type { Bundle } from "../../lib/schemas";
import { interpolatedPrecision, RECALL_GRID as GRID } from "../../lib/curves";
import { AXIS_TICK } from "../../lib/nav";
import { ChartCard, DataTable, LegendToggle, TooltipBox } from "./common";

export function PRCurves({ bundle, className }: { bundle: Bundle; className?: string }) {
  const served = bundle.meta.served_label;
  const order = useMemo(
    () => modelOrder(bundle).filter((m) => m in bundle.pr_curves.models),
    [bundle],
  );
  // Calibration is monotonic, so the served model's uncalibrated twin draws the identical
  // curve underneath it; start with the twin hidden.
  const [hidden, setHidden] = useState<Set<string>>(() => new Set([bundle.meta.served_model]));
  // Draw the served model last so it sits on top.
  const drawOrder = [...order.filter((m) => m !== served), served];

  const rows = useMemo(() => {
    const series = Object.fromEntries(
      order.map((m) => {
        const c = bundle.pr_curves.models[m];
        return [m, c ? interpolatedPrecision(c.recall, c.precision) : []];
      }),
    );
    return GRID.map((r, i) => ({
      recall: r,
      ...Object.fromEntries(order.map((m) => [m, series[m]?.[i] ?? null])),
    }));
  }, [bundle, order]);

  const toggle = (id: string) =>
    setHidden((h) => {
      const next = new Set(h);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const prevalence = bundle.pr_curves.prevalence;
  const legend = (
    <LegendToggle
      hidden={hidden}
      onToggle={toggle}
      items={order.map((m) => ({
        id: m,
        label: modelName(m),
        color: modelColor(m, served),
        note:
          m === bundle.meta.served_model
            ? "same curve as calibrated"
            : `AP ${fmt.num(metricValue(bundle, m, "pr_auc"))}`,
      }))}
    />
  );

  return (
    <ChartCard
      className={className}
      eyebrow="EV-2 · test slice"
      title="Precision–recall curves"
      description={`How many alerts are real fraud (precision) at each share of fraud caught (recall). The dotted line is a model with no skill: precision equal to the ${fmt.pct(prevalence, 3)} fraud rate.`}
      legend={legend}
      chart={
        <div className="h-[320px] w-full" role="img" aria-label="Precision-recall curves for each model on the test slice">
          <ResponsiveContainer>
            <LineChart data={rows} margin={{ top: 8, right: 12, bottom: 18, left: 0 }}>
              <CartesianGrid vertical={false} />
              <XAxis
                dataKey="recall"
                type="number"
                domain={[0, 1]}
                ticks={[0, 0.2, 0.4, 0.6, 0.8, 1]}
                tickFormatter={(v: number) => fmt.pct(v, 0)}
                tick={AXIS_TICK}
                label={{ value: "Recall (share of fraud caught)", position: "insideBottom", offset: -12 }}
              />
              <YAxis
                domain={[0, 1]}
                ticks={[0, 0.25, 0.5, 0.75, 1]}
                tickFormatter={(v: number) => fmt.pct(v, 0)}
                tick={AXIS_TICK}
                width={44}
              />
              <ReferenceLine y={prevalence} stroke="var(--ink-3)" strokeDasharray="2 4" />
              <Tooltip
                cursor={{ stroke: "var(--axis)", strokeWidth: 1 }}
                content={({ active, label, payload }) =>
                  active && payload?.length ? (
                    <TooltipBox
                      title={`Recall ${fmt.pct(Number(label), 1)}`}
                      rows={payload
                        .filter((p) => p.value != null)
                        .sort((a, b) => Number(b.value) - Number(a.value))
                        .map((p) => ({
                          label: modelName(String(p.dataKey)),
                          value: fmt.pct(Number(p.value), 1),
                          color: modelColor(String(p.dataKey), served),
                        }))}
                    />
                  ) : null
                }
              />
              {drawOrder.map((m) => (
                <Line
                  key={m}
                  dataKey={m}
                  hide={hidden.has(m)}
                  type="linear"
                  stroke={modelColor(m, served)}
                  strokeWidth={m === served ? 2.5 : 2}
                  dot={false}
                  activeDot={{ r: 4, strokeWidth: 2, stroke: "var(--surface)" }}
                  isAnimationActive={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      }
      table={
        <DataTable
          caption="Interpolated precision by recall"
          columns={[
            { key: "recall", label: "Recall", align: "right" },
            ...order.map((m) => ({ key: m, label: modelName(m), align: "right" as const })),
          ]}
          rows={rows
            .filter((_, i) => i % 10 === 0)
            .map((r) => ({
              recall: fmt.pct(r.recall, 0),
              ...Object.fromEntries(
                order.map((m) => [m, fmt.pct(Number((r as Record<string, number>)[m]), 1)]),
              ),
            }))}
        />
      }
    />
  );
}
