import * as Slider from "@radix-ui/react-slider";
import { RotateCcw, Target, TrendingDown } from "lucide-react";
import { useMemo, useState } from "react";
import { AnimatedNumber } from "../components/charts/common";
import { CostCurve } from "../components/charts/CostCurve";
import { ConfusionMatrix } from "../components/charts/misc";
import { PageHeader, Reveal } from "../components/layout";
import { Badge, Button, Callout, Card, CardHeader, InfoTip, Segmented } from "../components/ui";
import { WithBundle } from "../components/WithBundle";
import { fmt } from "../lib/format";
import type { Bundle } from "../lib/schemas";
import { minCostIndex, pointMetrics, type ReviewCosts } from "../lib/threshold";

type SplitName = "validation" | "test";

export function ThresholdExplorer() {
  return <WithBundle>{(bundle) => <Explorer bundle={bundle} />}</WithBundle>;
}

function Explorer({ bundle }: { bundle: Bundle }) {
  const [split, setSplit] = useState<SplitName>("validation");
  const sweep = bundle.sweeps[split];
  const [indexBySplit, setIndexBySplit] = useState<Record<SplitName, number>>({
    validation: bundle.sweeps.validation.chosen_index,
    test: bundle.sweeps.test.chosen_index,
  });
  const defaults: ReviewCosts = {
    falseAlarm: bundle.costs.config.false_alarm,
    caughtFraud: bundle.costs.config.caught_fraud,
  };
  const [costs, setCosts] = useState<ReviewCosts>(defaults);
  const index = indexBySplit[split];
  const setIndex = (i: number) => setIndexBySplit((s) => ({ ...s, [split]: i }));

  const point = useMemo(() => pointMetrics(sweep, index, costs), [sweep, index, costs]);
  const minIndex = useMemo(() => minCostIndex(sweep, costs), [sweep, costs]);
  const chosen = useMemo(
    () => pointMetrics(sweep, sweep.chosen_index, costs),
    [sweep, costs],
  );
  const best = useMemo(() => pointMetrics(sweep, minIndex, costs), [sweep, minIndex, costs]);
  const costsChanged =
    costs.falseAlarm !== defaults.falseAlarm || costs.caughtFraud !== defaults.caughtFraud;
  const last = sweep.threshold.length - 1;
  // At the served point show the served threshold itself (the sweep stores the lowest
  // flagged score there, which flags exactly the same transactions).
  const shownThreshold = index === sweep.chosen_index ? sweep.chosen_threshold : point.threshold;

  return (
    <div className="space-y-6">
      <PageHeader
        title="Threshold & cost explorer"
        description={
          <>
            Every alert costs a review; every missed fraud costs its amount. Move the threshold
            to see the trade-off. The served threshold ({fmt.threshold(bundle.meta.threshold)})
            is the one that minimised total cost on the <strong className="text-ink">validation</strong>{" "}
            slice.
          </>
        }
        actions={
          <Segmented<SplitName>
            label="Data slice"
            value={split}
            onChange={setSplit}
            options={[
              { value: "validation", label: `Validation · ${bundle.sweeps.validation.frauds} frauds` },
              { value: "test", label: `Test · ${bundle.sweeps.test.frauds} frauds` },
            ]}
          />
        }
      />

      {split === "test" && (
        <Callout tone="warning" title="Test view is illustrative">
          The threshold was fixed on validation before the test slice was read. Picking a
          threshold here would be tuning on test data, so this view only shows what a different
          choice would have cost.
        </Callout>
      )}

      <Card>
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <p className="eyebrow">Decision threshold</p>
            <p className="mt-1.5 text-[32px] font-semibold tracking-[-0.03em] text-ink">
              {fmt.threshold(shownThreshold)}
            </p>
            <p className="mt-1 text-[12px] text-ink-3">
              Flag a transaction when its fraud probability is at least this value.
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <Button
              size="sm"
              onClick={() => setIndex(sweep.chosen_index)}
              disabled={index === sweep.chosen_index}
            >
              <RotateCcw size={13} aria-hidden /> Served threshold
            </Button>
            <Button
              size="sm"
              variant="primary"
              onClick={() => setIndex(minIndex)}
              disabled={index === minIndex}
            >
              <TrendingDown size={13} aria-hidden /> Cheapest on {split}
            </Button>
          </div>
        </div>
        <div className="mt-6">
          <Slider.Root
            className="relative flex h-6 w-full touch-none select-none items-center"
            min={0}
            max={last}
            step={1}
            value={[index]}
            onValueChange={([v]) => v != null && setIndex(v)}
            aria-label="Decision threshold"
          >
            <Slider.Track className="relative h-1.5 grow overflow-hidden rounded-full bg-surface-3">
              <Slider.Range className="absolute h-full rounded-full bg-accent" />
            </Slider.Track>
            <Slider.Thumb
              aria-label="Decision threshold"
              aria-valuetext={`Threshold ${fmt.threshold(shownThreshold)}, ${fmt.int(point.flagged)} alerts`}
              className="block size-5 rounded-full border-2 border-[var(--surface)] bg-accent shadow-[0_0_0_4px_var(--accent-soft)] transition-transform hover:scale-110 focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-accent/40"
            />
            {minIndex === sweep.chosen_index ? (
              <Marker index={minIndex} last={last} label="served = cheapest" accent />
            ) : (
              <>
                <Marker index={sweep.chosen_index} last={last} label="served" />
                <Marker index={minIndex} last={last} label="cheapest" accent />
              </>
            )}
          </Slider.Root>
          <div className="mt-6 flex justify-between text-[11px] text-ink-3">
            <span>← stricter: fewer alerts</span>
            <span>looser: more alerts →</span>
          </div>
        </div>
      </Card>

      <Reveal>
        <div className="grid grid-cols-2 gap-4 lg:grid-cols-5">
          <Kpi label="Total cost" value={point.totalCost} format={fmt.money} sub={`missed ${fmt.money(point.missedCost)} + reviews ${fmt.money(point.reviewCost)}`} />
          <Kpi
            label="Saved vs flag nothing"
            value={point.savedVsNothing}
            format={fmt.money}
            sub={`${fmt.pctUnits(point.savedVsNothingPct)} of ${fmt.money(point.flagNothingCost)}`}
            tone={point.savedVsNothing >= 0 ? "good" : "critical"}
          />
          <Kpi label="Fraud caught (recall)" value={point.recall} format={(v) => fmt.pct(v, 1)} sub={`${fmt.int(point.tp)} of ${fmt.int(point.tp + point.fn)}`} />
          <Kpi label="Alerts that are fraud" value={point.precision} format={(v) => fmt.pct(v, 1)} sub="precision" />
          <Kpi label="Alerts raised" value={point.flagged} format={fmt.int} sub={`${fmt.pct(point.flagged / sweep.rows, 3)} of transactions`} />
        </div>
      </Reveal>

      <div className="grid gap-4 xl:grid-cols-3">
        <Card className="xl:col-span-2">
          <CardHeader
            eyebrow={`${split} slice`}
            title="Total cost by threshold"
            description="Click or drag on the chart to move the threshold. The curve uses the review costs on the right."
            actions={
              <Badge tone={index === minIndex ? "good" : "neutral"}>
                {index === minIndex ? "At minimum cost" : `${fmt.money(point.totalCost - best.totalCost)} above minimum`}
              </Badge>
            }
          />
          <CostCurve
            sweep={sweep}
            costs={costs}
            index={index}
            chosenIndex={sweep.chosen_index}
            onSelect={setIndex}
          />
        </Card>
        <Card>
          <CardHeader eyebrow="Outcome" title="Confusion matrix" description={`${fmt.int(sweep.rows)} ${split} transactions`} />
          <ConfusionMatrix tp={point.tp} fp={point.fp} fn={point.fn} tn={point.tn} />
          <dl className="mt-4 space-y-1.5 text-[12px]">
            <Row label="Missed-fraud cost" value={fmt.money(point.missedCost)} />
            <Row label="Review cost" value={fmt.money(point.reviewCost)} />
            <Row label="Flag everything would cost" value={fmt.money(point.flagEverythingCost)} />
          </dl>
        </Card>
      </div>

      <Card>
        <CardHeader
          eyebrow="Assumptions"
          title="Review costs"
          description="Change what a review costs and watch the cheapest threshold move. A missed fraud always costs its transaction amount; that comes from the data."
          actions={
            costsChanged && (
              <Button size="sm" variant="ghost" onClick={() => setCosts(defaults)}>
                <RotateCcw size={13} aria-hidden /> Reset to config.yaml
              </Button>
            )
          }
        />
        <div className="grid gap-4 md:grid-cols-3">
          <CostInput
            label="False alarm"
            hint="Reviewing a legitimate transaction"
            value={costs.falseAlarm}
            onChange={(v) => setCosts((c) => ({ ...c, falseAlarm: v }))}
          />
          <CostInput
            label="Caught fraud"
            hint="Reviewing a fraud that was flagged"
            value={costs.caughtFraud}
            onChange={(v) => setCosts((c) => ({ ...c, caughtFraud: v }))}
          />
          <div className="rounded-xl bg-surface-2 p-4 ring-1 ring-inset ring-line">
            <p className="eyebrow flex items-center gap-1.5">
              <Target size={12} aria-hidden /> Cheapest threshold on {split}
              <InfoTip content="Recomputed in your browser from the exported confusion counts. It is a what-if, not a re-fit." />
            </p>
            <p className="mt-2 text-[20px] font-semibold text-ink">{fmt.threshold(best.threshold)}</p>
            <p className="mt-1 text-[12px] text-ink-3">
              {fmt.money(best.totalCost)} total · served threshold costs{" "}
              {fmt.money(chosen.totalCost)}
            </p>
          </div>
        </div>
      </Card>
    </div>
  );
}

function Marker({ index, last, label, accent }: { index: number; last: number; label: string; accent?: boolean }) {
  return (
    <span
      className={`pointer-events-none absolute -translate-x-1/2 whitespace-nowrap text-[10px] font-medium ${accent ? "bottom-full mb-1" : "top-full mt-1"}`}
      style={{ left: `${(100 * index) / Math.max(last, 1)}%`, color: accent ? "var(--accent)" : "var(--ink-3)" }}
      aria-hidden
    >
      {accent ? `${label} ▼` : `▲ ${label}`}
    </span>
  );
}

function Kpi({
  label,
  value,
  format,
  sub,
  tone,
}: {
  label: string;
  value: number;
  format: (v: number) => string;
  sub?: string;
  tone?: "good" | "critical";
}) {
  return (
    <div className="card p-4">
      <p className="eyebrow">{label}</p>
      <AnimatedNumber
        value={value}
        format={format}
        className={`mt-2 block text-[22px] font-semibold tracking-[-0.02em] ${
          tone === "good" ? "text-good-ink" : tone === "critical" ? "text-critical-ink" : "text-ink"
        }`}
      />
      {sub && <p className="mt-1 text-[11px] text-ink-3">{sub}</p>}
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between">
      <dt className="text-ink-3">{label}</dt>
      <dd className="tnum text-ink">{value}</dd>
    </div>
  );
}

function CostInput({
  label,
  hint,
  value,
  onChange,
}: {
  label: string;
  hint: string;
  value: number;
  onChange: (v: number) => void;
}) {
  const id = `cost-${label.replace(/\s+/g, "-").toLowerCase()}`;
  return (
    <label htmlFor={id} className="block rounded-xl bg-surface-2 p-4 ring-1 ring-inset ring-line">
      <span className="eyebrow">{label}</span>
      <span className="mt-2 flex items-center gap-1 rounded-lg bg-surface px-3 ring-1 ring-inset ring-line-strong focus-within:ring-accent">
        <span className="text-ink-3">$</span>
        <input
          id={id}
          type="number"
          inputMode="decimal"
          min={0}
          max={10000}
          step={1}
          value={value}
          onChange={(e) => {
            const v = Number(e.target.value);
            if (Number.isFinite(v) && v >= 0 && v <= 10000) onChange(v);
          }}
          className="tnum h-9 w-full bg-transparent text-[15px] text-ink outline-none"
        />
      </span>
      <span className="mt-1.5 block text-[11px] text-ink-3">{hint}</span>
    </label>
  );
}
