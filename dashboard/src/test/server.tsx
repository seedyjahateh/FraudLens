import * as RadixTooltip from "@radix-ui/react-tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { vi } from "vitest";
import { App } from "../App";
import { ThemeProvider } from "../lib/theme";
import { BundleSchema, type Bundle } from "../lib/schemas";
import raw from "./fixture-bundle.json";

/** The bundle written by `fraudlens.evaluate` on the synthetic fixture (see README). */
export const fixtureBundle: Bundle = BundleSchema.parse(raw);

export const presets = {
  typical: Object.fromEntries(
    ["Time", ...Array.from({ length: 28 }, (_, i) => `V${i + 1}`), "Amount"].map((f) => [
      f,
      f === "Amount" ? 22 : f === "Time" ? 84000 : 0,
    ]),
  ),
  suspicious: Object.fromEntries(
    ["Time", ...Array.from({ length: 28 }, (_, i) => `V${i + 1}`), "Amount"].map((f) => [
      f,
      f === "V14" ? -4.29 : f === "Time" ? 406 : 0,
    ]),
  ),
};

type Handler = (init?: RequestInit) => { status?: number; body: unknown };

export function mockApi(overrides: Record<string, Handler> = {}) {
  const routes: Record<string, Handler> = {
    "/api/dashboard": () => ({ body: { ...raw, load_test: null } }),
    "/health": () => ({
      body: { status: "ok", model_loaded: true, model_version: fixtureBundle.meta.version },
    }),
    "/model": () => ({
      body: {
        model_name: fixtureBundle.meta.served_model,
        model_version: fixtureBundle.meta.version,
        trained_at: fixtureBundle.meta.trained_at,
        calibration: "sigmoid",
        threshold: fixtureBundle.meta.threshold,
        test_pr_auc: 0.9,
        test_pr_auc_ci: [0.8, 0.95],
        data_sha256: fixtureBundle.meta.data_sha256,
        code_version: fixtureBundle.meta.code_version,
        features: Object.keys(presets.typical),
      },
    }),
    "/api/stats": () => ({
      body: {
        started_at: "2026-10-06T12:00:00+00:00",
        uptime_seconds: 125,
        score_requests: 12,
        rejected_requests: 1,
        window: 11,
        p50_ms: 4.2,
        p95_ms: 9.8,
        p99_ms: 12.1,
        mean_ms: 5,
        flag_rate: 0.18,
        flagged: 2,
        allowed: 9,
        recent_ms: [3, 4, 5, 6, 4, 3, 9, 4, 5, 4, 3],
      },
    }),
    "/api/presets": () => ({ body: presets }),
    "/score": () => ({
      body: {
        request_id: "req-123456789",
        fraud_probability: 0.91,
        decision: "flag",
        threshold: fixtureBundle.meta.threshold,
        model_version: fixtureBundle.meta.version,
        top_features: [
          { feature: "V14", contribution: 0.6 },
          { feature: "V4", contribution: 0.12 },
          { feature: "Amount", contribution: -0.03 },
        ],
      },
    }),
    ...overrides,
  };
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input.toString();
    const path = url.split("?")[0] ?? url;
    const handler = routes[path];
    if (!handler) return new Response(JSON.stringify({ detail: "not found" }), { status: 404 });
    const { status = 200, body } = handler(init);
    return new Response(JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    });
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

export function renderApp(path = "/") {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false, refetchInterval: false } },
  });
  return render(
    <ThemeProvider>
      <QueryClientProvider client={client}>
        <RadixTooltip.Provider>
          <MemoryRouter initialEntries={[path]}>
            <App />
          </MemoryRouter>
        </RadixTooltip.Provider>
      </QueryClientProvider>
    </ThemeProvider>,
  );
}
