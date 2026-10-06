import { z } from "zod";

/*
 * Every API response is parsed with these schemas. A mismatch surfaces as an explicit
 * error state instead of a half-rendered page. They mirror fraudlens/dashboard_data.py
 * (bundle schema_version 1) and service/schemas.py.
 */

// prettier-ignore
export const FEATURES = [
  "Time",
  "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14",
  "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
  "Amount",
] as const;
export type Feature = (typeof FEATURES)[number];

const num = z.number();
const nullableNum = z.number().nullable();

export const MetricRowSchema = z.object({
  model: z.string(),
  role: z.enum(["served", "candidate", "baseline"]),
  split: z.enum(["validation", "test"]),
  metric: z.string(),
  value: num,
  ci_low: nullableNum,
  ci_high: nullableNum,
});
export type MetricRow = z.infer<typeof MetricRowSchema>;

export const ComparisonSchema = z.object({
  model_a: z.string(),
  model_b: z.string(),
  metric: z.string(),
  difference: num,
  ci_low: num,
  ci_high: num,
  significant: z.boolean(),
});
export type Comparison = z.infer<typeof ComparisonSchema>;

export const SweepSchema = z
  .object({
    threshold: z.array(num),
    tp: z.array(num),
    fp: z.array(num),
    fn: z.array(num),
    tn: z.array(num),
    missed_cost: z.array(num),
    chosen_index: z.number().int().nonnegative(),
    min_cost_index: z.number().int().nonnegative(),
    chosen_threshold: num,
    rows: num,
    frauds: num,
    fraud_amount: num,
  })
  .refine(
    (s) =>
      [s.tp, s.fp, s.fn, s.tn, s.missed_cost].every((a) => a.length === s.threshold.length) &&
      s.chosen_index < s.threshold.length &&
      s.min_cost_index < s.threshold.length,
    { message: "sweep arrays are inconsistent" },
  );
export type Sweep = z.infer<typeof SweepSchema>;

const CurveSchema = z.object({ recall: z.array(num), precision: z.array(num) });
const CalibrationSchema = z.object({
  predicted: z.array(num),
  observed: z.array(num),
  histogram: z.array(num),
});
const SliceSchema = z.object({ rows: num, frauds: num });
const TrialSchema = z.object({
  params: z.record(z.string(), z.unknown()),
  validation_pr_auc: num,
  seconds: num,
});

export const LoadTestSchema = z.object({
  requests: num,
  target: z.string(),
  environment: z.string(),
  p50_ms: num,
  p95_ms: num,
  p99_ms: num,
  max_ms: num,
  mean_ms: num.optional(),
  errors: num,
  budget_ms: num,
  measured_at: z.string(),
  model_version: z.string().optional(),
  payload_source: z.string().optional(),
});
export type LoadTest = z.infer<typeof LoadTestSchema>;

export const BundleSchema = z.object({
  schema_version: z.literal(1),
  meta: z.object({
    version: z.string(),
    data_sha256: z.string(),
    code_version: z.string(),
    seed: num,
    trained_at: z.string(),
    evaluated_at: z.string(),
    train_seconds: num,
    bootstrap_samples: num,
    confidence: num,
    served_model: z.string(),
    served_label: z.string(),
    calibration: z.string(),
    selection_rule: z.string(),
    threshold: num,
  }),
  split: z.object({ train: SliceSchema, validation: SliceSchema, test: SliceSchema }),
  metrics: z.array(MetricRowSchema),
  comparisons: z.array(ComparisonSchema),
  finding: z.object({ better_unselected: z.array(z.string()) }),
  pr_curves: z.object({ prevalence: num, models: z.record(z.string(), CurveSchema) }),
  sweeps: z.object({ validation: SweepSchema, test: SweepSchema }),
  calibration: z.record(z.string(), CalibrationSchema),
  costs: z.object({
    config: z.object({
      missed_fraud: z.union([z.literal("amount"), num]),
      false_alarm: num,
      caught_fraud: num,
    }),
    test: z.object({
      model_cost: num,
      flag_nothing_cost: num,
      flag_everything_cost: num,
    }),
    validation_outcome: z.object({
      threshold: num,
      total_cost: num,
      tp: num,
      fp: num,
      fn: num,
      tn: num,
    }),
  }),
  importance: z.array(
    z.object({ feature: z.string(), importance_mean: num, importance_std: num }),
  ),
  imbalance: z.array(
    z.object({ model: z.string(), strategy: z.string(), validation_pr_auc: num }),
  ),
  search: z.record(
    z.string(),
    z.object({
      params: z.record(z.string(), z.unknown()),
      validation_pr_auc: num,
      threshold: num,
      search_seconds: num,
      trials: z.array(TrialSchema),
    }),
  ),
  load_test: LoadTestSchema.nullable().optional(),
});
export type Bundle = z.infer<typeof BundleSchema>;

export const HealthSchema = z.object({
  status: z.literal("ok"),
  model_loaded: z.boolean(),
  model_version: z.string(),
});
export type Health = z.infer<typeof HealthSchema>;

export const ModelInfoSchema = z.object({
  model_name: z.string(),
  model_version: z.string(),
  trained_at: z.string(),
  calibration: z.string().nullable(),
  threshold: num,
  test_pr_auc: nullableNum,
  test_pr_auc_ci: z.array(num).nullable(),
  data_sha256: z.string(),
  code_version: z.string(),
  features: z.array(z.string()),
});
export type ModelInfo = z.infer<typeof ModelInfoSchema>;

export const StatsSchema = z.object({
  started_at: z.string(),
  uptime_seconds: num,
  score_requests: num,
  rejected_requests: num,
  window: num,
  p50_ms: nullableNum,
  p95_ms: nullableNum,
  p99_ms: nullableNum,
  mean_ms: nullableNum,
  flag_rate: nullableNum,
  flagged: num,
  allowed: num,
  recent_ms: z.array(num),
});
export type Stats = z.infer<typeof StatsSchema>;

export const TransactionSchema = z.object(
  Object.fromEntries(FEATURES.map((f) => [f, num])) as Record<Feature, typeof num>,
);
export type Transaction = Record<Feature, number>;

export const PresetsSchema = z.object({
  typical: TransactionSchema,
  suspicious: TransactionSchema,
});
export type Presets = z.infer<typeof PresetsSchema>;

export const ScoreResponseSchema = z.object({
  request_id: z.string(),
  fraud_probability: z.number().min(0).max(1),
  decision: z.enum(["flag", "allow"]),
  threshold: num,
  model_version: z.string(),
  top_features: z.array(z.object({ feature: z.string(), contribution: num })),
});
export type ScoreResponse = z.infer<typeof ScoreResponseSchema>;

export const ValidationErrorSchema = z.object({
  detail: z.string(),
  errors: z.array(z.object({ field: z.string(), message: z.string(), type: z.string() })),
});
export type ValidationErrorBody = z.infer<typeof ValidationErrorSchema>;
