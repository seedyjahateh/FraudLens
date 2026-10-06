import { Clock, Cpu, Database, GitCommit, Zap } from "lucide-react";
import type { ReactNode } from "react";
import { ChartCard, DataTable } from "../components/charts/common";
import { CalibrationChart, ImbalanceHeatmap, ImportanceBars, Sparkline } from "../components/charts/misc";
import { PageHeader, Reveal } from "../components/layout";
import { Badge, Card, CardHeader, CopyButton, ErrorState, InfoTip, Skeleton } from "../components/ui";
import { WithBundle } from "../components/WithBundle";
import { fmt, modelName } from "../lib/format";
import { useModel, useStats } from "../lib/queries";
import type { Bundle } from "../lib/schemas";

export function ModelService() {
  return <WithBundle>{(bundle) => <Content bundle={bundle} />}</WithBundle>;
}

function Content({ bundle }: { bundle: Bundle }) {
  return (
    <div className="space-y-6">
      <PageHeader
        title="Model & service"
        description="What is being served, where it came from, how it was chosen, and how the API is behaving right now."
      />
      <Reveal>
        <div className="grid gap-4 xl:grid-cols-3">
          <ModelCard />
          <LiveStats />
          <LoadTestCard bundle={bundle} />
        </div>
      </Reveal>
      <Reveal delay={0.05}>
        <div className="grid gap-4 xl:grid-cols-5">
          <SearchCard bundle={bundle} />
          <Card className="xl:col-span-2">
            <CardHeader
              eyebrow="EV-5 · validation PR-AUC"
              title="Imbalance strategies"
              description="Resampling happens inside the training fit only; validation and test rows are never resampled."
            />
            <ImbalanceHeatmap rows={bundle.imbalance} />
          </Card>
        </div>
      </Reveal>
      <Reveal delay={0.1}>
        <div className="grid gap-4 xl:grid-cols-2">
          <ChartCard
            eyebrow="FR-13 · validation slice"
            title="Permutation importance"
            description="Drop in validation PR-AUC when a column is shuffled. V1–V28 are anonymised PCA components: this says which component carries signal, not what it means."
            chart={<ImportanceBars rows={bundle.importance} />}
            table={
              <DataTable
                columns={[
                  { key: "feature", label: "Feature" },
                  { key: "mean", label: "Importance", align: "right" },
                  { key: "std", label: "± std", align: "right" },
                ]}
                rows={bundle.importance.map((r) => ({
                  feature: r.feature,
                  mean: r.importance_mean.toFixed(4),
                  std: r.importance_std.toFixed(4),
                }))}
              />
            }
          />
          <ChartCard
            eyebrow="EV-3 · test slice"
            title="Calibration"
            description={`Predicted probability against the observed fraud rate (dotted line = perfect). The served model is calibrated with ${bundle.meta.calibration} scaling on validation. Upper bins hold few transactions.`}
            chart={<CalibrationChart bundle={bundle} />}
            table={
              <DataTable
                columns={[
                  { key: "model", label: "Model" },
                  { key: "pred", label: "Predicted", align: "right" },
                  { key: "obs", label: "Observed", align: "right" },
                ]}
                rows={Object.entries(bundle.calibration).flatMap(([m, c]) =>
                  c.predicted.map((p, i) => ({
                    model: modelName(m),
                    pred: fmt.pct(p, 2),
                    obs: fmt.pct(c.observed[i] ?? NaN, 2),
                  })),
                )}
              />
            }
          />
        </div>
      </Reveal>
      <Reveal delay={0.15}>
        <SplitCard bundle={bundle} />
      </Reveal>
    </div>
  );
}

function Line({ icon, label, children }: { icon?: ReactNode; label: string; children: ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-3 border-b border-line py-2.5 text-[12.5px] last:border-0">
      <dt className="flex items-center gap-2 text-ink-3">
        {icon}
        {label}
      </dt>
      <dd className="flex min-w-0 items-center gap-1 text-right text-ink">{children}</dd>
    </div>
  );
}

function ModelCard() {
  const { data, error, isPending, isError, refetch } = useModel();
  return (
    <Card>
      <CardHeader eyebrow="GET /model" title="Served model" />
      {isPending ? (
        <Skeleton className="h-56" />
      ) : isError ? (
        <ErrorState error={error} onRetry={() => void refetch()} />
      ) : (
        <dl>
          <Line icon={<Cpu size={13} aria-hidden />} label="Model">
            {modelName(data.model_name)}
            {data.calibration && <Badge tone="accent">{data.calibration}</Badge>}
          </Line>
          <Line label="Version">
            <span className="font-mono text-[12px]">{data.model_version}</span>
          </Line>
          <Line icon={<Clock size={13} aria-hidden />} label="Trained">
            {fmt.date(data.trained_at)}
          </Line>
          <Line label="Threshold">{fmt.threshold(data.threshold)}</Line>
          <Line label="Test PR-AUC">
            {data.test_pr_auc != null ? fmt.num(data.test_pr_auc) : "not evaluated"}
            {data.test_pr_auc_ci && (
              <span className="tnum text-ink-3">
                ({fmt.ci(data.test_pr_auc_ci[0] ?? null, data.test_pr_auc_ci[1] ?? null)})
              </span>
            )}
          </Line>
          <Line icon={<Database size={13} aria-hidden />} label="Data SHA-256">
            <span className="truncate font-mono text-[12px]" title={data.data_sha256}>
              {fmt.shortHash(data.data_sha256)}
            </span>
            <CopyButton value={data.data_sha256} label="data hash" />
          </Line>
          <Line icon={<GitCommit size={13} aria-hidden />} label="Code version">
            <span className="truncate font-mono text-[12px]" title={data.code_version}>
              {fmt.shortHash(data.code_version)}
            </span>
            <CopyButton value={data.code_version} label="code version" />
          </Line>
          <Line label="Inputs">{data.features.length} features</Line>
        </dl>
      )}
    </Card>
  );
}

function LiveStats() {
  const { data, error, isPending, isError, refetch, isFetching } = useStats();
  return (
    <Card>
      <CardHeader
        eyebrow="GET /api/stats · every 5 s"
        title="Live service"
        description="Rolling window of the most recent /score calls served by this process."
        actions={isFetching ? <span className="size-1.5 animate-pulse rounded-full bg-accent" aria-hidden /> : null}
      />
      {isPending ? (
        <Skeleton className="h-56" />
      ) : isError ? (
        <ErrorState error={error} onRetry={() => void refetch()} />
      ) : (
        <>
          <div className="grid grid-cols-3 gap-2">
            <Mini label="p50" value={fmt.ms(data.p50_ms)} />
            <Mini label="p95" value={fmt.ms(data.p95_ms)} strong />
            <Mini label="p99" value={fmt.ms(data.p99_ms)} />
          </div>
          <div className="mt-4 rounded-xl bg-surface-2 p-3 ring-1 ring-inset ring-line">
            <p className="eyebrow mb-2">Latency, last {data.recent_ms.length} requests</p>
            <Sparkline values={data.recent_ms} />
          </div>
          <dl className="mt-3">
            <Line label="Scored requests">{fmt.int(data.score_requests)}</Line>
            <Line label="Rejected (422)">{fmt.int(data.rejected_requests)}</Line>
            <Line label="Flag rate">
              {data.flag_rate != null ? fmt.pct(data.flag_rate, 1) : "–"}
              <InfoTip content="Share of recent scored requests flagged. Depends entirely on what was sent; it is not a fraud rate." />
            </Line>
            <Line label="Uptime">{fmt.duration(data.uptime_seconds)}</Line>
          </dl>
        </>
      )}
    </Card>
  );
}

function Mini({ label, value, strong }: { label: string; value: string; strong?: boolean }) {
  return (
    <div className="rounded-xl bg-surface-2 px-3 py-2.5 ring-1 ring-inset ring-line">
      <p className="eyebrow">{label}</p>
      <p className={`mt-1 text-[18px] tracking-[-0.02em] ${strong ? "font-semibold text-ink" : "text-ink-2"}`}>
        {value}
      </p>
    </div>
  );
}

function LoadTestCard({ bundle }: { bundle: Bundle }) {
  const lt = bundle.load_test;
  return (
    <Card>
      <CardHeader eyebrow="NFR-2 · scripts/load_test.py" title="Load test" />
      {!lt ? (
        <p className="text-[13px] text-ink-3">
          No load test recorded for model {bundle.meta.version}. Run{" "}
          <code className="font-mono text-ink-2">python scripts/load_test.py</code> against the running API.
        </p>
      ) : (
        <>
          <div className="flex items-end gap-3">
            <p className="text-[40px] font-semibold leading-none tracking-[-0.04em] text-ink">
              {lt.p95_ms.toFixed(1)}
              <span className="text-[16px] font-medium text-ink-3"> ms p95</span>
            </p>
            <Badge tone={lt.p95_ms < lt.budget_ms ? "good" : "critical"}>
              <Zap size={12} aria-hidden />
              {lt.p95_ms < lt.budget_ms ? "within" : "over"} {lt.budget_ms} ms budget
            </Badge>
          </div>
          <div className="mt-5 h-2 overflow-hidden rounded-full bg-surface-3" aria-hidden>
            <div
              className="h-full rounded-full bg-accent"
              style={{ width: `${Math.min(100, (100 * lt.p95_ms) / lt.budget_ms)}%` }}
            />
          </div>
          <dl className="mt-4">
            <Line label="Requests">{fmt.int(lt.requests)} sequential</Line>
            <Line label="p50 / p99">
              {fmt.ms(lt.p50_ms)} / {fmt.ms(lt.p99_ms)}
            </Line>
            <Line label="Max">{fmt.ms(lt.max_ms)}</Line>
            <Line label="Errors">{fmt.int(lt.errors)}</Line>
            <Line label="Measured">{lt.measured_at}</Line>
          </dl>
          <p className="mt-3 text-[11px] leading-relaxed text-ink-3">{lt.environment}. Client-side, end to end.</p>
        </>
      )}
    </Card>
  );
}

function SearchCard({ bundle }: { bundle: Bundle }) {
  const served = bundle.meta.served_model;
  return (
    <Card className="xl:col-span-3">
      <CardHeader
        eyebrow="FR-8 · randomised search on validation"
        title="How the model was chosen"
        description={`Each setting is trained on the training slice and scored by validation PR-AUC. Selection rule: ${bundle.meta.selection_rule}.`}
      />
      <div className="-mx-2 overflow-x-auto">
        <table className="w-full min-w-[560px] text-[12.5px]">
          <caption className="sr-only">Search results per model</caption>
          <thead>
            <tr className="text-left text-ink-3">
              <th scope="col" className="px-2 pb-2 font-medium">Model</th>
              <th scope="col" className="px-2 pb-2 text-right font-medium">Tried</th>
              <th scope="col" className="px-2 pb-2 text-right font-medium">Best val PR-AUC</th>
              <th scope="col" className="px-2 pb-2 font-medium">Best parameters</th>
              <th scope="col" className="px-2 pb-2 text-right font-medium">Time</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(bundle.search).map(([name, s]) => (
              <tr key={name} className="border-t border-line align-top">
                <th scope="row" className="px-2 py-2.5 text-left font-medium text-ink">
                  {modelName(name)} {name === served && <Badge tone="accent">selected</Badge>}
                </th>
                <td className="tnum px-2 py-2.5 text-right text-ink-2">{s.trials.length}</td>
                <td className="tnum px-2 py-2.5 text-right text-ink">
                  {s.validation_pr_auc.toFixed(4)}
                  <TrialDots trials={s.trials.map((t) => t.validation_pr_auc)} best={s.validation_pr_auc} />
                </td>
                <td className="px-2 py-2.5">
                  <div className="flex flex-wrap gap-1">
                    {Object.entries(s.params)
                      .sort(([a], [b]) => a.localeCompare(b))
                      .map(([k, v]) => (
                        <span key={k} className="rounded bg-surface-2 px-1.5 py-0.5 font-mono text-[11px] text-ink-2 ring-1 ring-inset ring-line">
                          {k}={String(v)}
                        </span>
                      ))}
                  </div>
                </td>
                <td className="tnum whitespace-nowrap px-2 py-2.5 text-right text-ink-3">{fmt.duration(s.search_seconds)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

/** Each trial as a dot on a 0..1 line: the spread shows how much tuning mattered. */
function TrialDots({ trials, best }: { trials: number[]; best: number }) {
  return (
    <span className="relative mt-1.5 block h-2 w-full min-w-24 rounded-full bg-surface-3" aria-hidden>
      {trials.map((t, i) => (
        <span
          key={i}
          className="absolute top-1/2 size-2 -translate-x-1/2 -translate-y-1/2 rounded-full ring-1 ring-[var(--surface)]"
          style={{ left: `${t * 100}%`, background: t === best ? "var(--series-1)" : "var(--axis)" }}
        />
      ))}
    </span>
  );
}

function SplitCard({ bundle }: { bundle: Bundle }) {
  const parts = [
    { name: "Train", ...bundle.split.train, note: "fit models, scalers, samplers" },
    { name: "Validation", ...bundle.split.validation, note: "search, calibration, threshold" },
    { name: "Test", ...bundle.split.test, note: "read once, for reporting" },
  ];
  const total = parts.reduce((s, p) => s + p.rows, 0);
  const colors = ["var(--series-1)", "var(--series-3)", "var(--series-2)"];
  return (
    <Card>
      <CardHeader
        eyebrow="EV-1 · chronological split"
        title="Train → validation → test, in time order"
        description="Sorted by Time and cut 60/20/20, each cut moved to the next distinct timestamp. Random splits are not used for headline numbers because they leak future patterns into training."
      />
      <div className="flex h-3 gap-0.5 overflow-hidden rounded-full" aria-hidden>
        {parts.map((p, i) => (
          <span key={p.name} style={{ width: `${(100 * p.rows) / total}%`, background: colors[i] }} />
        ))}
      </div>
      <div className="mt-4 grid gap-3 sm:grid-cols-3">
        {parts.map((p, i) => (
          <div key={p.name} className="rounded-xl bg-surface-2 p-4 ring-1 ring-inset ring-line">
            <p className="flex items-center gap-2 text-[13px] font-medium text-ink">
              <span className="h-2 w-3.5 rounded-full" style={{ background: colors[i] }} aria-hidden />
              {p.name}
            </p>
            <p className="tnum mt-2 text-[20px] font-semibold text-ink">{fmt.int(p.rows)}</p>
            <p className="text-[12px] text-ink-3">
              {fmt.int(p.frauds)} frauds · {fmt.pct(p.frauds / p.rows, 3)}
            </p>
            <p className="mt-2 text-[11px] text-ink-3">{p.note}</p>
          </div>
        ))}
      </div>
    </Card>
  );
}
