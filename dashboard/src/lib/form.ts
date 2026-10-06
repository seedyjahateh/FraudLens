import { ApiError } from "./api";
import { FEATURES, type Feature, type Transaction } from "./schemas";

export type FormValues = Record<Feature, string>;
export type FieldErrors = Partial<Record<Feature, string>>;

export const PCA_FEATURES = FEATURES.filter((f) => f.startsWith("V"));

const round = (v: number) => Math.round(v * 10000) / 10000;

export const toForm = (t: Transaction): FormValues =>
  Object.fromEntries(FEATURES.map((f) => [f, String(round(t[f]))])) as FormValues;

/** Mirrors the API's rules so most mistakes are caught before a request is sent. */
export function validateForm(values: FormValues): {
  transaction?: Transaction;
  errors: FieldErrors;
} {
  const errors: FieldErrors = {};
  const out: Partial<Transaction> = {};
  for (const f of FEATURES) {
    const raw = values[f].trim();
    if (raw === "") {
      errors[f] = "Required";
      continue;
    }
    const v = Number(raw);
    if (!Number.isFinite(v)) errors[f] = "Must be a number";
    else if ((f === "Amount" || f === "Time") && v < 0) errors[f] = "Must be ≥ 0";
    else out[f] = v;
  }
  return Object.keys(errors).length ? { errors } : { transaction: out as Transaction, errors };
}

/** Map a 422 body from the API onto form fields. */
export function serverFieldErrors(error: unknown): FieldErrors {
  if (!(error instanceof ApiError) || !error.validation) return {};
  const out: FieldErrors = {};
  for (const e of error.validation.errors) {
    if ((FEATURES as readonly string[]).includes(e.field)) out[e.field as Feature] = e.message;
  }
  return out;
}

function gaussian(): number {
  const u = 1 - Math.random();
  const v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** A plausible random transaction around the typical one (never a dataset row). */
export function randomise(base: Transaction): Transaction {
  const t = { ...base };
  for (const f of PCA_FEATURES) t[f] = round(base[f] + gaussian() * 1.1);
  t.Amount = round(Math.max(0, Math.exp(Math.log(Math.max(base.Amount, 1)) + gaussian())));
  t.Time = Math.round(Math.random() * 172_000);
  return t;
}
