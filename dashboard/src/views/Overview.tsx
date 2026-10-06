import { ArrowRight, ShieldCheck } from "lucide-react";
import { Link } from "react-router";
import { ChartCard, DataTable } from "../components/charts/common";
import { ForestPlot } from "../components/charts/ForestPlot";
import { ConfusionMatrix } from "../components/charts/misc";
import { PRCurves } from "../components/charts/PRCurves";
import { PageHeader, Reveal } from "../components/layout";
import { Badge, Callout, Card, CardHeader, Stat } from "../components/ui";
import { WithBundle } from "../components/WithBundle";
import { fmt, modelName } from "../lib/format";
import {
  metric,
  metricValue,
  modelColor,
  modelOrder,
  targetLabel,
  targetMetrics,
} from "../lib/models";
import type { Bundle } from "../lib/schemas";

export function Overview() {
  return <WithBundle>{(bundle) => <OverviewContent bundle={bundle} />}</WithBundle>;
}

function ci(bundle: Bundle, model: string, name: string, digits = 3): string {
  const row = metric(bundle, model, name);
  return row?.ci_low != null && row.ci_high != null
    ? `95% CI ${fmt.ci(row.ci_low, row.ci_high, digits)}`
    : "";
}

function OverviewContent({ bundle }: { bundle: Bundle }) {
  const served = bundle.meta.served_label;
  const { recallAtP, precisionAtR } = targetMetrics(bundle);
  const tp = metricValue(bundle, served, "tp");
  const fn = metricValue(bundle, served, "fn");
  const nothing = bundle.costs.test.flag_nothing_cost;
  const cost = bundle.costs.test.model_cost;
  const savedPct = metric(bundle, served, "savings_vs_flag_nothing_pct");
  const order = modelOrder(bundle);
  const better = bundle.finding.better_unselected;

  return (
    <div className="space-y-6">
      <PageHeader
        title="Results on the held-out test slice"
        description={
          <>
            The final 20% of transactions by time, read once after the model, imbalance strategy,
            calibration and threshold were all fixed on validation data. Intervals are 95%
            bootstrap intervals over {fmt.int(bundle.meta.bootstrap_samples)} resamples.
          </>
        }
        actions={
          <div className="flex flex-wrap gap-2">
            <Badge tone="accent">
              <ShieldCheck size={12} aria-hidden /> Served: {modelName(served)}
            </Badge>
            <Badge>
              <span className="font-mono">{bundle.meta.version}</span>
            </Badge>
          </div>
        }
      />

      {better.length > 0 && (
        <Reveal>
          <Callout tone="warning" title="The model chosen on validation is not the best on test">
            {better.map(modelName).join(", ")} scored significantly higher PR-AUC on the test
            slice than the served model (paired bootstrap interval below zero). The served model
            is kept on purpose: it was selected before the test slice was read, and switching
            now would be choosing a model on test data. With only {bundle.split.validation.frauds}{" "}
            validation frauds, model selection is itself noisy.
          </Callout>
        </Reveal>
      )}

      <Reveal delay={0.05}>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-5">
          <Stat
            label="Test PR-AUC"
            value={fmt.num(metricValue(bundle, served, "pr_auc"))}
            sub={ci(bundle, served, "pr_auc")}
            info="Average precision: area under the precision–recall curve. Unlike accuracy or ROC-AUC, it is not flattered by the 99.8% legitimate majority."
          />
          <Stat
            label={`Recall @ ${targetLabel(recallAtP)} precision`}
            value={fmt.pct(metricValue(bundle, served, recallAtP), 1)}
            sub={ci(bundle, served, recallAtP)}
            info="Share of fraud caught when at least 90% of alerts must be real fraud."
          />
          <Stat
            label="Frauds caught"
            value={
              <>
                {fmt.int(tp)}
                <span className="text-[18px] font-medium text-ink-3"> / {fmt.int(tp + fn)}</span>
              </>
            }
            sub={`${fmt.int(metricValue(bundle, served, "fp"))} false alarms at threshold ${fmt.threshold(bundle.meta.threshold)}`}
          />
          <Stat
            label="Saved vs flag nothing"
            value={fmt.money(nothing - cost)}
            tone="good"
            sub={
              savedPct
                ? `${fmt.pctUnits(savedPct.value)} · CI ${fmt.pctUnits(savedPct.ci_low ?? NaN, 0)}–${fmt.pctUnits(savedPct.ci_high ?? NaN, 0)}`
                : undefined
            }
            info="Missed fraud costs its amount; every alert costs a $5 review. Money figures describe this dataset only."
          />
          <Stat
            label="API latency p95"
            value={bundle.load_test ? fmt.ms(bundle.load_test.p95_ms) : "—"}
            sub={
              bundle.load_test
                ? `${fmt.int(bundle.load_test.requests)} requests · budget ${bundle.load_test.budget_ms} ms`
                : "No load test recorded for this version"
            }
          />
        </div>
      </Reveal>

      <Reveal delay={0.1}>
        <div className="grid gap-4 xl:grid-cols-5">
          <PRCurves bundle={bundle} className="xl:col-span-3" />
          <Card className="xl:col-span-2">
            <CardHeader
              eyebrow="EV-7 · decision"
              title={`At the cost-chosen threshold (${fmt.threshold(bundle.meta.threshold)})`}
              description={`Chosen on validation to minimise total cost, then applied unchanged to the ${fmt.int(bundle.split.test.rows)} test transactions.`}
            />
            <ConfusionMatrix
              tp={tp}
              fn={fn}
              fp={metricValue(bundle, served, "fp")}
              tn={metricValue(bundle, served, "tn")}
            />
            <div className="mt-4 grid grid-cols-3 gap-2 text-[12px]">
              {[
                { label: "Model", value: cost, strong: true },
                { label: "Flag nothing", value: nothing },
                { label: "Flag everything", value: bundle.costs.test.flag_everything_cost },
              ].map((c) => (
                <div key={c.label} className="rounded-lg bg-surface-2 px-3 py-2 ring-1 ring-inset ring-line">
                  <p className="text-ink-3">{c.label}</p>
                  <p className={`tnum mt-0.5 text-[14px] ${c.strong ? "font-semibold text-ink" : "text-ink-2"}`}>
                    {fmt.money(c.value)}
                  </p>
                </div>
              ))}
            </div>
          </Card>
        </div>
      </Reveal>

      <Reveal delay={0.15}>
        <div className="grid gap-4 xl:grid-cols-2">
          <ChartCard
            eyebrow="EV-4 · uncertainty"
            title="Test PR-AUC with 95% intervals"
            description="Overlapping intervals are not evidence of a difference; the paired comparison on the right is the proper test."
            chart={
              <ForestPlot
                ariaLabel="Test PR-AUC with confidence intervals for each model"
                domain={[0, 1]}
                ticks={[0, 0.25, 0.5, 0.75, 1]}
                format={(v) => v.toFixed(2)}
                rows={order.map((m) => {
                  const r = metric(bundle, m, "pr_auc");
                  return {
                    id: m,
                    label: modelName(m),
                    value: r?.value ?? NaN,
                    low: r?.ci_low ?? NaN,
                    high: r?.ci_high ?? NaN,
                    color: modelColor(m, served),
                    emphasis: m === served,
                  };
                })}
              />
            }
            table={
              <DataTable
                columns={[
                  { key: "model", label: "Model" },
                  { key: "value", label: "PR-AUC", align: "right" },
                  { key: "ci", label: "95% CI", align: "right" },
                ]}
                rows={order.map((m) => {
                  const r = metric(bundle, m, "pr_auc");
                  return {
                    model: modelName(m),
                    value: fmt.num(r?.value ?? NaN),
                    ci: fmt.ci(r?.ci_low ?? null, r?.ci_high ?? null),
                  };
                })}
              />
            }
          />
          <ChartCard
            eyebrow="Paired bootstrap"
            title="Served model minus each alternative"
            description="Difference in test PR-AUC on the same resampled rows. Significant only when the interval excludes zero."
            chart={
              <ForestPlot
                ariaLabel="Paired PR-AUC differences between the served model and each alternative"
                domain={symmetricDomain(bundle)}
                reference={0}
                separator=" to "
                ticks={niceTicks(symmetricDomain(bundle))}
                format={(v) => fmt.signed(v, 2)}
                rows={bundle.comparisons.map((c) => ({
                  id: c.model_b,
                  label: `vs ${modelName(c.model_b)}`,
                  value: c.difference,
                  low: c.ci_low,
                  high: c.ci_high,
                  color: modelColor(c.model_b, served),
                  emphasis: c.significant,
                  note: c.significant ? "significant" : "not significant",
                }))}
              />
            }
            table={
              <DataTable
                columns={[
                  { key: "vs", label: "Compared with" },
                  { key: "diff", label: "Difference", align: "right" },
                  { key: "ci", label: "95% CI", align: "right" },
                  { key: "sig", label: "Verdict" },
                ]}
                rows={bundle.comparisons.map((c) => ({
                  vs: modelName(c.model_b),
                  diff: fmt.signed(c.difference),
                  ci: `${fmt.signed(c.ci_low)} to ${fmt.signed(c.ci_high)}`,
                  sig: c.significant ? "Significant" : "Not significant",
                }))}
              />
            }
          />
        </div>
      </Reveal>

      <Reveal delay={0.2}>
        <div className="grid gap-4 xl:grid-cols-3">
          <AccuracyParadox bundle={bundle} />
          <Card className="xl:col-span-2">
            <CardHeader
              eyebrow="All models · test slice"
              title="Model comparison"
              description={`Validation PR-AUC chose the model; everything else is test. ROC-AUC is shown only to illustrate how it flatters rare-event models.`}
              actions={
                <Link
                  to="/threshold"
                  className="inline-flex items-center gap-1 text-[12px] font-medium text-accent hover:underline"
                >
                  Explore the threshold <ArrowRight size={13} aria-hidden />
                </Link>
              }
            />
            <div className="-mx-2 overflow-x-auto">
              <table className="w-full min-w-[640px] text-[12.5px]">
                <caption className="sr-only">Test metrics for every model with 95% intervals</caption>
                <thead>
                  <tr className="text-left text-ink-3">
                    <th scope="col" className="px-2 pb-2 font-medium">Model</th>
                    <th scope="col" className="px-2 pb-2 text-right font-medium">Val PR-AUC</th>
                    <th scope="col" className="px-2 pb-2 text-right font-medium">Test PR-AUC</th>
                    <th scope="col" className="px-2 pb-2 text-right font-medium">Recall @{targetLabel(recallAtP)} P</th>
                    <th scope="col" className="px-2 pb-2 text-right font-medium">Precision @{targetLabel(precisionAtR)} R</th>
                    <th scope="col" className="px-2 pb-2 text-right font-medium">ROC-AUC</th>
                    <th scope="col" className="px-2 pb-2 text-right font-medium">Brier</th>
                  </tr>
                </thead>
                <tbody>
                  {order.map((m) => (
                    <tr key={m} className="border-t border-line">
                      <th scope="row" className="px-2 py-2.5 text-left font-medium text-ink">
                        <span className="flex items-center gap-2">
                          <span className="h-2 w-3.5 rounded-full" style={{ background: modelColor(m, served) }} aria-hidden />
                          {modelName(m)}
                          {m === served && <Badge tone="accent">served</Badge>}
                        </span>
                      </th>
                      <td className="tnum px-2 text-right text-ink-2">{fmt.num(metricValue(bundle, m, "pr_auc", "validation"))}</td>
                      {["pr_auc", recallAtP, precisionAtR, "roc_auc", "brier"].map((name) => {
                        const r = metric(bundle, m, name);
                        const digits = name === "brier" ? 4 : 3;
                        return (
                          <td key={name} className="tnum px-2 text-right">
                            <span className="text-ink">{fmt.num(r?.value ?? NaN, digits)}</span>
                            <span className="block whitespace-nowrap text-[11px] text-ink-3">
                              {fmt.ci(r?.ci_low ?? null, r?.ci_high ?? null, digits)}
                            </span>
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      </Reveal>
    </div>
  );
}

function symmetricDomain(bundle: Bundle): [number, number] {
  const extent = Math.max(
    0.05,
    ...bundle.comparisons.flatMap((c) => [Math.abs(c.ci_low), Math.abs(c.ci_high)]),
  );
  const nice = Math.ceil(extent * 10) / 10;
  return [-nice, nice];
}

function niceTicks([lo, hi]: [number, number]): number[] {
  const step = (hi - lo) / 4;
  return [lo, lo + step, 0, hi - step, hi];
}

function AccuracyParadox({ bundle }: { bundle: Bundle }) {
  const served = bundle.meta.served_label;
  const legitAcc = metricValue(bundle, "always_legit", "accuracy");
  const servedAcc = metricValue(bundle, served, "accuracy");
  const servedRecall = metricValue(bundle, served, "recall_at_threshold");
  const rows = [
    { label: "Always “legitimate”", accuracy: legitAcc, recall: 0, color: "var(--axis)" },
    { label: modelName(served), accuracy: servedAcc, recall: servedRecall, color: "var(--series-1)" },
  ];
  const bar = (v: number, color: string) => (
    <div className="h-2 flex-1 overflow-hidden rounded-full bg-surface-3">
      <div className="h-full rounded-full" style={{ width: `${v * 100}%`, background: color }} />
    </div>
  );
  return (
    <Card>
      <CardHeader
        eyebrow="Why not accuracy"
        title="The accuracy paradox"
        description="A model that never flags anything is almost perfectly accurate here. Recall tells them apart."
      />
      <div className="space-y-5">
        {rows.map((r) => (
          <div key={r.label}>
            <p className="mb-2 text-[13px] font-medium text-ink">{r.label}</p>
            <div className="space-y-1.5 text-[12px]">
              <div className="flex items-center gap-3">
                <span className="w-16 text-ink-3">Accuracy</span>
                {bar(r.accuracy, r.color)}
                <span className="tnum w-14 text-right text-ink">{fmt.pct(r.accuracy, 2)}</span>
              </div>
              <div className="flex items-center gap-3">
                <span className="w-16 text-ink-3">Recall</span>
                {bar(r.recall, r.color)}
                <span className="tnum w-14 text-right text-ink">{fmt.pct(r.recall, 0)}</span>
              </div>
            </div>
          </div>
        ))}
      </div>
      <p className="mt-5 text-[12px] leading-relaxed text-ink-3">
        Accuracy differs by {fmt.num((servedAcc - legitAcc) * 100, 2)} percentage points; recall
        by {fmt.pct(servedRecall, 0)}.
      </p>
    </Card>
  );
}
