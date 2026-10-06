import { Tip } from "../ui";

export interface ForestRow {
  id: string;
  label: string;
  value: number;
  low: number;
  high: number;
  color: string;
  emphasis?: boolean;
  note?: string;
}

/**
 * Point estimates with confidence intervals on one shared axis. Optional reference line
 * (e.g. 0 for a difference). Built in HTML so it stays crisp and responsive.
 */
export function ForestPlot({
  rows,
  domain,
  reference,
  format,
  ticks,
  ariaLabel,
  separator = "–",
}: {
  rows: ForestRow[];
  domain: [number, number];
  reference?: number;
  format: (v: number) => string;
  ticks: number[];
  ariaLabel: string;
  separator?: string;
}) {
  const [lo, hi] = domain;
  const pos = (v: number) => `${(100 * (Math.min(Math.max(v, lo), hi) - lo)) / (hi - lo)}%`;
  return (
    <div role="img" aria-label={ariaLabel} className="w-full">
      <div className="space-y-1">
        {rows.map((r) => (
          <div key={r.id} className="group grid grid-cols-[1fr_auto] items-center gap-x-3 sm:grid-cols-[minmax(120px,190px)_1fr_auto]">
            <span
              className={`col-span-2 truncate pt-1 text-[12px] sm:col-span-1 sm:pt-0 ${r.emphasis ? "font-semibold text-ink" : "text-ink-2"}`}
              title={r.label}
            >
              {r.label}
            </span>
            <Tip
              content={
                <span className="tnum">
                  {r.label}: {format(r.value)} (95% CI {format(r.low)} – {format(r.high)})
                  {r.note ? ` · ${r.note}` : ""}
                </span>
              }
            >
              <div
                tabIndex={0}
                className="relative h-8 rounded-md transition-colors group-hover:bg-surface-2 focus-visible:bg-surface-2"
              >
                {ticks.map((t) => (
                  <span
                    key={t}
                    className="absolute inset-y-1 w-px bg-grid"
                    style={{ left: pos(t) }}
                    aria-hidden
                  />
                ))}
                {reference != null && (
                  <span
                    className="absolute inset-y-0 w-px bg-ink-3"
                    style={{ left: pos(reference) }}
                    aria-hidden
                  />
                )}
                <span
                  className="absolute top-1/2 h-[3px] -translate-y-1/2 rounded-full opacity-60"
                  style={{
                    left: pos(r.low),
                    width: `calc(${pos(r.high)} - ${pos(r.low)})`,
                    background: r.color,
                  }}
                  aria-hidden
                />
                <span
                  className="absolute top-1/2 size-3 -translate-x-1/2 -translate-y-1/2 rounded-full ring-2 ring-[var(--surface)]"
                  style={{ left: pos(r.value), background: r.color }}
                  aria-hidden
                />
              </div>
            </Tip>
            <span className="tnum min-w-[128px] whitespace-nowrap text-right text-[12px] text-ink-2">
              <span className="font-medium text-ink">{format(r.value)}</span>{" "}
              <span className="text-ink-3">
                {format(r.low)}{separator}{format(r.high)}
              </span>
            </span>
          </div>
        ))}
      </div>
      <div className="mt-1 grid grid-cols-[1fr_auto] gap-x-3 sm:grid-cols-[minmax(120px,190px)_1fr_auto]">
        <span className="hidden sm:block" />
        <div className="relative h-5">
          {ticks.map((t) => (
            <span
              key={t}
              className="tnum absolute -translate-x-1/2 text-[11px] text-ink-3"
              style={{ left: pos(t) }}
            >
              {format(t)}
            </span>
          ))}
        </div>
        <span className="min-w-[128px]" />
      </div>
    </div>
  );
}
