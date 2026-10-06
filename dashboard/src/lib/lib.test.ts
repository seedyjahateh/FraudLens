import { afterEach, describe, expect, it, vi } from "vitest";
import { fixtureBundle } from "../test/server";
import { api, ApiError } from "./api";
import { interpolatedPrecision } from "./curves";
import { fmt, modelName } from "./format";
import { serverFieldErrors, toForm, validateForm } from "./form";
import { metricValue, modelColor, modelOrder, targetMetrics } from "./models";
import { BundleSchema } from "./schemas";
import {
  baselines,
  indexForThreshold,
  minCostIndex,
  pointMetrics,
  totalCost,
  type ReviewCosts,
} from "./threshold";

const configCosts: ReviewCosts = {
  falseAlarm: fixtureBundle.costs.config.false_alarm,
  caughtFraud: fixtureBundle.costs.config.caught_fraud,
};

describe("threshold math matches the numbers Python wrote", () => {
  const val = fixtureBundle.sweeps.validation;
  const test = fixtureBundle.sweeps.test;

  it("reproduces the validation cost at the served threshold", () => {
    const p = pointMetrics(val, val.chosen_index, configCosts);
    const expected = fixtureBundle.costs.validation_outcome;
    expect(p.tp).toBe(expected.tp);
    expect(p.fp).toBe(expected.fp);
    expect(p.fn).toBe(expected.fn);
    expect(p.totalCost).toBeCloseTo(expected.total_cost, 1);
  });

  it("reproduces the test cost and the baselines", () => {
    const p = pointMetrics(test, test.chosen_index, configCosts);
    expect(p.totalCost).toBeCloseTo(fixtureBundle.costs.test.model_cost, 1);
    const b = baselines(test, configCosts);
    expect(b.nothing).toBeCloseTo(fixtureBundle.costs.test.flag_nothing_cost, 1);
    expect(b.everything).toBeCloseTo(fixtureBundle.costs.test.flag_everything_cost, 1);
    expect(p.tp).toBe(metricValue(fixtureBundle, fixtureBundle.meta.served_label, "tp"));
  });

  it("finds the same minimum-cost point", () => {
    expect(minCostIndex(val, configCosts)).toBe(val.min_cost_index);
    expect(minCostIndex(test, configCosts)).toBe(test.min_cost_index);
  });

  it("maps the served threshold back to its sweep point", () => {
    expect(indexForThreshold(val, val.chosen_threshold)).toBe(val.chosen_index);
    expect(indexForThreshold(test, test.chosen_threshold)).toBe(test.chosen_index);
  });

  it("moves the cheapest threshold when reviews get expensive", () => {
    const pricey = { falseAlarm: 500, caughtFraud: 500 };
    const cheapIdx = minCostIndex(val, configCosts);
    const priceyIdx = minCostIndex(val, pricey);
    const flagged = (i: number) => (val.tp[i] ?? 0) + (val.fp[i] ?? 0);
    expect(flagged(priceyIdx)).toBeLessThanOrEqual(flagged(cheapIdx));
    expect(totalCost(val, 0, pricey)).toBe(val.missed_cost[0]);
  });

  it("derives precision, recall and savings consistently", () => {
    const p = pointMetrics(test, test.chosen_index, configCosts);
    expect(p.precision).toBeCloseTo(p.tp / Math.max(p.flagged, 1));
    expect(p.recall).toBeCloseTo(p.tp / (p.tp + p.fn));
    expect(p.savedVsNothing).toBeCloseTo(p.flagNothingCost - p.totalCost);
    expect(p.tp + p.fp + p.fn + p.tn).toBe(test.rows);
  });
});

describe("schemas", () => {
  it("accept the bundle evaluate writes", () => {
    expect(BundleSchema.safeParse(fixtureBundle).success).toBe(true);
  });

  it("reject an inconsistent sweep", () => {
    const broken = structuredClone(fixtureBundle);
    broken.sweeps.test.tp = broken.sweeps.test.tp.slice(1);
    expect(BundleSchema.safeParse(broken).success).toBe(false);
  });

  it("reject an unknown schema version", () => {
    expect(BundleSchema.safeParse({ ...fixtureBundle, schema_version: 2 }).success).toBe(false);
  });
});

describe("models", () => {
  it("serves first and keeps colours fixed per entity", () => {
    const order = modelOrder(fixtureBundle);
    expect(order[0]).toBe(fixtureBundle.meta.served_label);
    expect(modelColor("random_forest", fixtureBundle.meta.served_label)).toBe("var(--series-3)");
    expect(modelColor(fixtureBundle.meta.served_label, fixtureBundle.meta.served_label)).toBe(
      "var(--series-1)",
    );
  });

  it("finds the configured target metrics", () => {
    expect(targetMetrics(fixtureBundle)).toEqual({
      recallAtP: "recall_at_90pct_precision",
      precisionAtR: "precision_at_80pct_recall",
    });
  });
});

describe("format", () => {
  it("formats numbers for people", () => {
    expect(fmt.money(3241.4)).toBe("$3,241");
    expect(fmt.pct(0.7333)).toBe("73.3%");
    expect(fmt.threshold(0.30317)).toBe("0.303");
    expect(fmt.threshold(0.00004)).toBe("4.0e-5");
    expect(fmt.threshold(1.2)).toBe("> max score");
    expect(fmt.signed(-0.03)).toBe("−0.030");
    expect(fmt.ms(4.24)).toBe("4.2 ms");
    expect(fmt.ms(null)).toBe("–");
    expect(fmt.duration(125)).toBe("2 min 5 s");
    expect(fmt.shortHash("76274b691b16a6c49d3f159c883398e0")).toBe("76274b69…98e0");
    expect(modelName("hist_gradient_boosting (calibrated)")).toBe("Gradient boosting · calibrated");
  });
});

describe("curves", () => {
  it("interpolates precision as a non-increasing step", () => {
    const out = interpolatedPrecision([1, 0.5, 0.5, 0], [0.4, 0.5, 1, 1], [0, 0.25, 0.5, 0.75, 1]);
    expect(out).toEqual([1, 1, 1, 0.4, 0.4]);
  });
});

describe("playground form", () => {
  const base = toForm(
    Object.fromEntries(
      ["Time", ...Array.from({ length: 28 }, (_, i) => `V${i + 1}`), "Amount"].map((f) => [f, 1]),
    ) as never,
  );

  it("accepts valid input", () => {
    expect(validateForm(base).transaction?.Amount).toBe(1);
  });

  it("rejects blanks, text and negative amounts", () => {
    const { transaction, errors } = validateForm({ ...base, V3: "", V5: "abc", Amount: "-2" });
    expect(transaction).toBeUndefined();
    expect(errors).toEqual({ V3: "Required", V5: "Must be a number", Amount: "Must be ≥ 0" });
  });

  it("maps a 422 body onto fields", () => {
    const err = new ApiError("invalid transaction", {
      status: 422,
      kind: "http",
      validation: {
        detail: "invalid transaction",
        errors: [
          { field: "Amount", message: "Input should be greater than or equal to 0", type: "x" },
          { field: "nonsense", message: "ignored", type: "x" },
        ],
      },
    });
    expect(serverFieldErrors(err)).toEqual({ Amount: "Input should be greater than or equal to 0" });
    expect(serverFieldErrors(new Error("boom"))).toEqual({});
  });
});

describe("api client", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("turns a 422 into a typed validation error", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        new Response(
          JSON.stringify({
            detail: "invalid transaction",
            errors: [{ field: "Amount", message: "too small", type: "greater_than_equal" }],
          }),
          { status: 422 },
        ),
      ),
    );
    const err = await api.score({} as never).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).status).toBe(422);
    expect((err as ApiError).validation?.errors[0]?.field).toBe("Amount");
  });

  it("flags a contract mismatch as a schema error", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ status: "nope" }))));
    const err = await api.health().catch((e: unknown) => e);
    expect((err as ApiError).kind).toBe("schema");
  });

  it("reports an unreachable service as a network error", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => Promise.reject(new TypeError("failed"))));
    const err = await api.model().catch((e: unknown) => e);
    expect((err as ApiError).kind).toBe("network");
  });
});
