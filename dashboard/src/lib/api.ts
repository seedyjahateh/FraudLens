import type { z } from "zod";
import {
  BundleSchema,
  HealthSchema,
  ModelInfoSchema,
  PresetsSchema,
  ScoreResponseSchema,
  StatsSchema,
  ValidationErrorSchema,
  type Transaction,
  type ValidationErrorBody,
} from "./schemas";

/** An HTTP or contract failure, with enough detail for an honest error state. */
export class ApiError extends Error {
  readonly status: number;
  readonly kind: "http" | "network" | "schema";
  readonly validation?: ValidationErrorBody;

  constructor(
    message: string,
    opts: { status?: number; kind: ApiError["kind"]; validation?: ValidationErrorBody },
  ) {
    super(message);
    this.name = "ApiError";
    this.status = opts.status ?? 0;
    this.kind = opts.kind;
    this.validation = opts.validation;
  }
}

async function request<T>(
  path: string,
  schema: z.ZodType<T>,
  init?: RequestInit,
): Promise<{ data: T; ms: number }> {
  const started = performance.now();
  let response: Response;
  try {
    response = await fetch(path, {
      ...init,
      headers: { Accept: "application/json", ...init?.headers },
    });
  } catch {
    throw new ApiError("The scoring service is unreachable.", { kind: "network" });
  }
  const ms = performance.now() - started;
  const body: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    const validation = ValidationErrorSchema.safeParse(body);
    const detail =
      validation.success
        ? validation.data.detail
        : typeof body === "object" && body !== null && "detail" in body
          ? String((body as { detail: unknown }).detail)
          : response.statusText;
    throw new ApiError(detail || `Request failed (${response.status})`, {
      status: response.status,
      kind: "http",
      validation: validation.success ? validation.data : undefined,
    });
  }
  const parsed = schema.safeParse(body);
  if (!parsed.success) {
    const issue = parsed.error.issues[0];
    const where = issue?.path.join(".") || "response";
    throw new ApiError(`Unexpected response from ${path} (${where}: ${issue?.message})`, {
      status: response.status,
      kind: "schema",
    });
  }
  return { data: parsed.data, ms };
}

export const api = {
  bundle: () => request("/api/dashboard", BundleSchema).then((r) => r.data),
  health: () => request("/health", HealthSchema).then((r) => r.data),
  model: () => request("/model", ModelInfoSchema).then((r) => r.data),
  stats: () => request("/api/stats", StatsSchema).then((r) => r.data),
  presets: () => request("/api/presets", PresetsSchema).then((r) => r.data),
  score: (transaction: Transaction) =>
    request("/score", ScoreResponseSchema, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(transaction),
    }),
};
