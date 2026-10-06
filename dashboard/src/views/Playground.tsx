import clsx from "clsx";
import { Ban, CheckCircle2, Dices, History, Loader2, Send, ShieldAlert, Trash2 } from "lucide-react";
import { motion } from "motion/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router";
import { toast } from "sonner";
import { ContributionBars, ProbabilityGauge } from "../components/charts/misc";
import { PageHeader } from "../components/layout";
import { Badge, Button, Card, CardHeader, ErrorState, Kbd, Skeleton } from "../components/ui";
import { fmt } from "../lib/format";
import {
  PCA_FEATURES as PCA,
  randomise,
  serverFieldErrors,
  toForm,
  validateForm,
  type FieldErrors,
  type FormValues,
} from "../lib/form";
import { usePresets, useScore } from "../lib/queries";
import type { Feature, ScoreResponse, Transaction } from "../lib/schemas";
import { storage } from "../lib/storage";

interface HistoryEntry {
  at: string;
  label: string;
  transaction: Transaction;
  probability: number;
  decision: "flag" | "allow";
}

type PresetName = "typical" | "suspicious";
const HISTORY_KEY = "fraudlens.playground.history";
const PRESET_LABEL: Record<PresetName, string> = {
  typical: "Typical transaction",
  suspicious: "Suspicious example",
};

export function Playground() {
  const presets = usePresets();
  const [params] = useSearchParams();
  const raw = params.get("preset");
  const deepLink: PresetName | null = raw === "typical" || raw === "suspicious" ? raw : null;
  if (presets.isPending)
    return (
      <div className="space-y-6">
        <Skeleton className="h-9 w-72" />
        <div className="grid gap-4 xl:grid-cols-5">
          <Skeleton className="h-[560px] xl:col-span-3" />
          <Skeleton className="h-[560px] xl:col-span-2" />
        </div>
      </div>
    );
  if (presets.isError)
    return <ErrorState error={presets.error} onRetry={() => void presets.refetch()} />;
  // A deep link (?preset=…) remounts the playground with that preset and scores it.
  return (
    <PlaygroundContent
      key={deepLink ?? "default"}
      typical={presets.data.typical}
      suspicious={presets.data.suspicious}
      deepLink={deepLink}
    />
  );
}

function PlaygroundContent({
  typical,
  suspicious,
  deepLink,
}: {
  typical: Transaction;
  suspicious: Transaction;
  deepLink: PresetName | null;
}) {
  const start: PresetName = deepLink ?? "suspicious";
  const startTx = start === "typical" ? typical : suspicious;
  const [values, setValues] = useState<FormValues>(() => toForm(startTx));
  const [label, setLabel] = useState(PRESET_LABEL[start]);
  const [errors, setErrors] = useState<FieldErrors>({});
  const [result, setResult] = useState<{ data: ScoreResponse; ms: number } | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>(() => storage.get(HISTORY_KEY, []));
  const score = useScore();
  const formRef = useRef<HTMLFormElement>(null);

  /** Send a validated transaction; all state updates happen in the async callbacks. */
  const run = useCallback(
    (transaction: Transaction, name: string) => {
      score.mutate(transaction, {
        onSuccess: (res) => {
          setResult(res);
          setErrors({});
          setHistory((h) => {
            const next = [
              {
                at: new Date().toISOString(),
                label: name,
                transaction,
                probability: res.data.fraud_probability,
                decision: res.data.decision,
              },
              ...h,
            ].slice(0, 12);
            storage.set(HISTORY_KEY, next);
            return next;
          });
        },
        onError: (err) => {
          setErrors(serverFieldErrors(err));
          toast.error(err instanceof Error ? err.message : "Scoring failed");
        },
      });
    },
    [score],
  );

  const submit = (vals: FormValues, name: string) => {
    const { transaction, errors: clientErrors } = validateForm(vals);
    setErrors(clientErrors);
    if (!transaction) {
      toast.error("Fix the highlighted fields before scoring.");
      return;
    }
    run(transaction, name);
  };

  const load = (t: Transaction, name: string, autoscore = true) => {
    setValues(toForm(t));
    setLabel(name);
    setErrors({});
    if (autoscore) run(t, name);
  };

  useEffect(() => {
    if (deepLink) run(startTx, PRESET_LABEL[deepLink]);
    // Run once per mount; the key on this component changes with the deep link.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const set = (f: Feature, v: string) => {
    setValues((s) => ({ ...s, [f]: v }));
    setLabel("Custom");
    if (errors[f]) setErrors((e) => ({ ...e, [f]: undefined }));
  };

  const errorCount = useMemo(() => Object.values(errors).filter(Boolean).length, [errors]);
  return (
    <div className="space-y-6">
      <PageHeader
        title="Scoring playground"
        description="Send a transaction to the live /score endpoint and see the probability, the decision at the served threshold and the five inputs that moved the score most. No data is stored on the server."
      />
      <div className="grid gap-4 xl:grid-cols-5">
        <Card className="xl:col-span-3">
          <CardHeader
            eyebrow="Transaction"
            title={label}
            description="Presets never come from the dataset: “typical” is the training median of every input; “suspicious” is hand-made."
            actions={
              <div className="flex flex-wrap gap-1.5">
                <Button size="sm" onClick={() => load(typical, "Typical transaction")}>Typical</Button>
                <Button size="sm" onClick={() => load(suspicious, "Suspicious example")}>
                  <ShieldAlert size={13} aria-hidden /> Suspicious
                </Button>
                <Button size="sm" onClick={() => load(randomise(typical), "Randomised")}>
                  <Dices size={13} aria-hidden /> Randomise
                </Button>
              </div>
            }
          />
          <form
            ref={formRef}
            noValidate
            onSubmit={(e) => {
              e.preventDefault();
              submit(values, label);
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) formRef.current?.requestSubmit();
            }}
          >
            <div className="grid gap-3 sm:grid-cols-2">
              <Field feature="Amount" prefix="$" hint="Transaction amount" value={values.Amount} error={errors.Amount} onChange={set} large />
              <Field feature="Time" suffix="s" hint={`Seconds since first transaction · hour ${Math.floor((Number(values.Time) || 0) / 3600) % 24}`} value={values.Time} error={errors.Time} onChange={set} large />
            </div>
            <p className="eyebrow mb-2 mt-5">PCA components V1–V28</p>
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-4 lg:grid-cols-7">
              {PCA.map((f) => (
                <Field key={f} feature={f} value={values[f]} error={errors[f]} onChange={set} />
              ))}
            </div>
            <div className="mt-5 flex flex-wrap items-center justify-between gap-3 border-t border-line pt-4">
              <p className="text-[12px] text-ink-3" aria-live="polite">
                {errorCount > 0 ? (
                  <span className="text-critical-ink">{errorCount} field{errorCount > 1 ? "s" : ""} need attention</span>
                ) : (
                  <>
                    Press <Kbd>Ctrl</Kbd> <Kbd>Enter</Kbd> to score
                  </>
                )}
              </p>
              <Button type="submit" variant="primary" disabled={score.isPending}>
                {score.isPending ? <Loader2 size={14} className="animate-spin" aria-hidden /> : <Send size={14} aria-hidden />}
                Score transaction
              </Button>
            </div>
          </form>
        </Card>

        <div className="space-y-4 xl:col-span-2">
          <Card>
            <CardHeader eyebrow="Result" title="Model decision" />

              {result ? (
                <motion.div
                  key={result.data.request_id}
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.25 }}
                  className={clsx(score.isPending && "opacity-60 transition-opacity")}
                >
                  <ProbabilityGauge probability={result.data.fraud_probability} threshold={result.data.threshold} />
                  <div className="mt-4 flex justify-center">
                    <Decision decision={result.data.decision} />
                  </div>
                  <p className="eyebrow mb-3 mt-6">Top contributing inputs</p>
                  <ContributionBars items={result.data.top_features} />
                  <dl className="mt-5 grid grid-cols-3 gap-2 text-[11px]">
                    <Meta label="Round trip" value={fmt.ms(result.ms)} />
                    <Meta label="Model" value={result.data.model_version} mono />
                    <Meta label="Request" value={result.data.request_id.slice(0, 8)} mono />
                  </dl>
                </motion.div>
              ) : (
                <div className="flex h-64 flex-col items-center justify-center gap-2 text-center text-[13px] text-ink-3">
                  <Send size={20} aria-hidden />
                  Score a transaction to see the decision.
                </div>
              )}

          </Card>

          <Card>
            <CardHeader
              eyebrow="This browser only"
              title={
                <span className="flex items-center gap-2">
                  <History size={14} aria-hidden /> Recent scores
                </span>
              }
              actions={
                history.length > 0 && (
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => {
                      setHistory([]);
                      storage.set(HISTORY_KEY, []);
                    }}
                  >
                    <Trash2 size={13} aria-hidden /> Clear
                  </Button>
                )
              }
            />
            {history.length === 0 ? (
              <p className="text-[12px] text-ink-3">Nothing yet.</p>
            ) : (
              <ul className="-mx-2 space-y-0.5">
                {history.map((h) => (
                  <li key={h.at}>
                    <button
                      type="button"
                      onClick={() => load(h.transaction, h.label, false)}
                      className="flex w-full items-center gap-3 rounded-lg px-2 py-2 text-left text-[12px] hover:bg-surface-2"
                    >
                      <span
                        className={clsx("size-2 shrink-0 rounded-full", h.decision === "flag" ? "bg-critical" : "bg-good")}
                        aria-hidden
                      />
                      <span className="truncate text-ink">{h.label}</span>
                      <span className="tnum ml-auto text-ink-2">{fmt.pct(h.probability, 2)}</span>
                      <span className="w-12 text-right text-ink-3">{h.decision}</span>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
}

function Decision({ decision }: { decision: "flag" | "allow" }) {
  return decision === "flag" ? (
    <Badge tone="critical" className="!px-3 !py-1 !text-[13px]">
      <Ban size={14} aria-hidden /> Flag for review
    </Badge>
  ) : (
    <Badge tone="good" className="!px-3 !py-1 !text-[13px]">
      <CheckCircle2 size={14} aria-hidden /> Allow
    </Badge>
  );
}

function Meta({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="rounded-lg bg-surface-2 px-2.5 py-2 ring-1 ring-inset ring-line">
      <dt className="text-ink-3">{label}</dt>
      <dd className={clsx("mt-0.5 truncate text-ink", mono && "font-mono")}>{value}</dd>
    </div>
  );
}

function Field({
  feature,
  value,
  error,
  onChange,
  hint,
  prefix,
  suffix,
  large,
}: {
  feature: Feature;
  value: string;
  error?: string;
  onChange: (f: Feature, v: string) => void;
  hint?: string;
  prefix?: string;
  suffix?: string;
  large?: boolean;
}) {
  const id = `field-${feature}`;
  return (
    <div>
      <label htmlFor={id} className={clsx("mb-1 block font-mono text-ink-2", large ? "text-[12px]" : "text-[11px]")}>
        {feature}
      </label>
      <div
        className={clsx(
          "flex items-center rounded-lg bg-surface-2 ring-1 ring-inset transition-shadow focus-within:ring-2",
          error ? "ring-critical focus-within:ring-critical" : "ring-line focus-within:ring-accent",
        )}
      >
        {prefix && <span className="pl-2.5 text-[12px] text-ink-3">{prefix}</span>}
        <input
          id={id}
          name={feature}
          inputMode="decimal"
          autoComplete="off"
          spellCheck={false}
          value={value}
          aria-invalid={Boolean(error)}
          aria-describedby={error ? `${id}-error` : undefined}
          onChange={(e) => onChange(feature, e.target.value)}
          className={clsx(
            "tnum w-full min-w-0 bg-transparent px-2.5 text-ink outline-none",
            large ? "h-10 text-[15px]" : "h-8 text-[12px]",
          )}
        />
        {suffix && <span className="pr-2.5 text-[12px] text-ink-3">{suffix}</span>}
      </div>
      {error ? (
        <p id={`${id}-error`} className="mt-1 text-[11px] text-critical-ink">
          {error}
        </p>
      ) : (
        hint && <p className="mt-1 text-[11px] text-ink-3">{hint}</p>
      )}
    </div>
  );
}
