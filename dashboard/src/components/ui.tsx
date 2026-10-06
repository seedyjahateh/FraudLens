import * as RadixTooltip from "@radix-ui/react-tooltip";
import clsx from "clsx";
import { AlertTriangle, Check, Copy, Info, RefreshCw, ServerCrash } from "lucide-react";
import { useState, type ButtonHTMLAttributes, type ReactNode } from "react";
import { ApiError } from "../lib/api";

export function Card({
  children,
  className,
  as: Tag = "section",
  ...rest
}: {
  children: ReactNode;
  className?: string;
  as?: "section" | "div" | "article";
} & React.HTMLAttributes<HTMLElement>) {
  return (
    <Tag className={clsx("card min-w-0 p-5 sm:p-6", className)} {...rest}>
      {children}
    </Tag>
  );
}

export function CardHeader({
  eyebrow,
  title,
  description,
  actions,
}: {
  eyebrow?: string;
  title: ReactNode;
  description?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <header className="mb-5 flex flex-wrap items-start justify-between gap-3">
      <div className="min-w-0 max-w-2xl">
        {eyebrow && <p className="eyebrow mb-1.5">{eyebrow}</p>}
        <h2 className="text-[15px] font-semibold tracking-[-0.01em] text-ink">{title}</h2>
        {description && <p className="mt-1 text-[13px] leading-relaxed text-ink-2">{description}</p>}
      </div>
      {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
    </header>
  );
}

type Tone = "neutral" | "accent" | "good" | "warning" | "critical";

const toneClass: Record<Tone, string> = {
  neutral: "bg-surface-3 text-ink-2 ring-line",
  accent: "bg-accent-soft text-accent ring-accent/25",
  good: "bg-good/12 text-good-ink ring-good/25",
  warning: "bg-warning/15 text-warning-ink ring-warning/30",
  critical: "bg-critical/12 text-critical-ink ring-critical/25",
};

export function Badge({
  children,
  tone = "neutral",
  className,
}: {
  children: ReactNode;
  tone?: Tone;
  className?: string;
}) {
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium ring-1 ring-inset",
        toneClass[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

export function Button({
  variant = "secondary",
  size = "md",
  className,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "ghost";
  size?: "sm" | "md";
}) {
  return (
    <button
      type="button"
      className={clsx(
        "inline-flex items-center justify-center gap-1.5 rounded-lg font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50",
        size === "sm" ? "h-7 px-2.5 text-[12px]" : "h-9 px-3.5 text-[13px]",
        variant === "primary" && "bg-accent text-white shadow-sm hover:brightness-110",
        variant === "secondary" &&
          "bg-surface-2 text-ink ring-1 ring-inset ring-line hover:bg-surface-3",
        variant === "ghost" && "text-ink-2 hover:bg-surface-3 hover:text-ink",
        className,
      )}
      {...props}
    />
  );
}

export function Skeleton({ className }: { className?: string }) {
  return <div className={clsx("animate-pulse rounded-lg bg-surface-3", className)} aria-hidden />;
}

export function Segmented<T extends string>({
  value,
  options,
  onChange,
  label,
}: {
  value: T;
  options: { value: T; label: ReactNode }[];
  onChange: (value: T) => void;
  label: string;
}) {
  return (
    <div
      role="radiogroup"
      aria-label={label}
      className="inline-flex rounded-lg bg-surface-2 p-0.5 ring-1 ring-inset ring-line"
    >
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={o.value === value}
          onClick={() => onChange(o.value)}
          className={clsx(
            "rounded-md px-3 py-1 text-[12px] font-medium transition-all",
            o.value === value
              ? "bg-surface text-ink shadow-sm ring-1 ring-line"
              : "text-ink-3 hover:text-ink",
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Callout({
  tone = "neutral",
  title,
  children,
  icon,
}: {
  tone?: Exclude<Tone, "accent"> | "accent";
  title: ReactNode;
  children?: ReactNode;
  icon?: ReactNode;
}) {
  const Icon = tone === "warning" || tone === "critical" ? AlertTriangle : Info;
  return (
    <div
      role={tone === "warning" || tone === "critical" ? "note" : undefined}
      className={clsx(
        "flex gap-3 rounded-xl p-4 ring-1 ring-inset",
        tone === "warning" && "bg-warning/8 ring-warning/30",
        tone === "critical" && "bg-critical/8 ring-critical/30",
        tone === "accent" && "bg-accent-soft ring-accent/25",
        (tone === "neutral" || tone === "good") && "bg-surface-2 ring-line",
      )}
    >
      <span
        className={clsx(
          "mt-0.5 shrink-0",
          tone === "warning" && "text-warning-ink",
          tone === "critical" && "text-critical-ink",
          tone === "accent" && "text-accent",
          (tone === "neutral" || tone === "good") && "text-ink-3",
        )}
      >
        {icon ?? <Icon size={16} aria-hidden />}
      </span>
      <div className="min-w-0 text-[13px] leading-relaxed">
        <p className="font-semibold text-ink">{title}</p>
        {children && <div className="mt-1 text-ink-2">{children}</div>}
      </div>
    </div>
  );
}

export function ErrorState({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  const api = error instanceof ApiError ? error : null;
  const title =
    api?.kind === "network"
      ? "Can't reach the scoring service"
      : api?.kind === "schema"
        ? "The service returned data in an unexpected shape"
        : api?.status === 404
          ? "Nothing to show yet"
          : "Something went wrong";
  return (
    <div className="flex flex-col items-start gap-3 rounded-xl bg-surface-2 p-5 ring-1 ring-inset ring-line">
      <ServerCrash size={20} className="text-critical-ink" aria-hidden />
      <div>
        <p className="text-[14px] font-semibold text-ink">{title}</p>
        <p className="mt-1 max-w-prose text-[13px] text-ink-2">
          {error instanceof Error ? error.message : String(error)}
        </p>
      </div>
      {onRetry && (
        <Button size="sm" onClick={onRetry}>
          <RefreshCw size={13} aria-hidden /> Retry
        </Button>
      )}
    </div>
  );
}

export function CopyButton({ value, label }: { value: string; label: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      aria-label={`Copy ${label}`}
      onClick={() => {
        void navigator.clipboard
          ?.writeText(value)
          .then(() => {
            setCopied(true);
            window.setTimeout(() => setCopied(false), 1400);
          })
          .catch(() => undefined);
      }}
      className="inline-flex size-6 items-center justify-center rounded-md text-ink-3 transition-colors hover:bg-surface-3 hover:text-ink"
    >
      {copied ? <Check size={13} className="text-good-ink" /> : <Copy size={13} />}
    </button>
  );
}

export function Kbd({ children }: { children: ReactNode }) {
  return (
    <kbd className="rounded border border-line-strong bg-surface-2 px-1.5 py-0.5 font-mono text-[10px] text-ink-3">
      {children}
    </kbd>
  );
}

export function Tip({ content, children }: { content: ReactNode; children: ReactNode }) {
  return (
    <RadixTooltip.Root delayDuration={150}>
      <RadixTooltip.Trigger asChild>{children}</RadixTooltip.Trigger>
      <RadixTooltip.Portal>
        <RadixTooltip.Content
          sideOffset={6}
          className="z-50 max-w-xs rounded-lg bg-ink px-2.5 py-1.5 text-[12px] leading-snug text-page shadow-lg"
        >
          {content}
          <RadixTooltip.Arrow className="fill-ink" />
        </RadixTooltip.Content>
      </RadixTooltip.Portal>
    </RadixTooltip.Root>
  );
}

export function InfoTip({ content }: { content: ReactNode }) {
  return (
    <Tip content={content}>
      <button
        type="button"
        aria-label="More information"
        className="inline-flex size-4 items-center justify-center rounded-full text-ink-3 hover:text-ink"
      >
        <Info size={12} aria-hidden />
      </button>
    </Tip>
  );
}

/** A labelled number with an optional interval and footnote. Hero figures stay proportional. */
export function Stat({
  label,
  value,
  sub,
  tone,
  info,
}: {
  label: ReactNode;
  value: ReactNode;
  sub?: ReactNode;
  tone?: "good" | "critical";
  info?: ReactNode;
}) {
  return (
    <div className="card relative overflow-hidden p-5">
      <div className="flex items-center gap-1.5">
        <p className="eyebrow">{label}</p>
        {info && <InfoTip content={info} />}
      </div>
      <p
        className={clsx(
          "mt-3 text-[28px] font-semibold leading-none tracking-[-0.03em]",
          tone === "good" && "text-good-ink",
          tone === "critical" && "text-critical-ink",
          !tone && "text-ink",
        )}
      >
        {value}
      </p>
      {sub && <p className="mt-2.5 text-[12px] text-ink-3">{sub}</p>}
    </div>
  );
}
