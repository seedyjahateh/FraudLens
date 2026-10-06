import { Command } from "cmdk";
import clsx from "clsx";
import { Command as CommandIcon, FlaskConical, Gauge, Moon, Search, Sun } from "lucide-react";
import { motion } from "motion/react";
import { Component, useEffect, useState, type ErrorInfo, type ReactNode } from "react";
import { NavLink, useLocation, useNavigate } from "react-router";
import { NAV } from "../lib/nav";
import { useHealth } from "../lib/queries";
import { useTheme } from "../lib/theme";
import { Button, ErrorState, Kbd, Tip } from "./ui";



function Brand() {
  return (
    <div className="flex items-center gap-2.5 px-2">
      <div className="relative flex size-8 items-center justify-center rounded-[10px] bg-gradient-to-br from-[var(--series-1)] to-[#1c5cab] shadow-[0_6px_20px_-6px_var(--series-1)]">
        <Search size={15} strokeWidth={2.5} className="text-white" aria-hidden />
        <span className="absolute right-1.5 top-1.5 size-1.5 rounded-full bg-[#ff8a8a]" />
      </div>
      <div className="leading-tight">
        <p className="text-[14px] font-semibold tracking-[-0.01em] text-ink">FraudLens</p>
        <p className="text-[11px] text-ink-3">Honest fraud detection</p>
      </div>
    </div>
  );
}

function HealthDot() {
  const { data, isError, isPending } = useHealth();
  const state = isPending ? "checking" : isError || !data ? "offline" : "live";
  return (
    <Tip
      content={
        state === "live"
          ? `Service healthy · model ${data?.model_version}`
          : state === "offline"
            ? "The scoring service is not responding"
            : "Checking the service…"
      }
    >
      <span
        className="inline-flex items-center gap-2 rounded-full bg-surface-2 px-2.5 py-1 text-[12px] text-ink-2 ring-1 ring-inset ring-line"
        role="status"
        aria-live="polite"
      >
        <span className="relative flex size-2">
          {state === "live" && (
            <span className="absolute inline-flex size-full animate-ping rounded-full bg-good opacity-50" />
          )}
          <span
            className={clsx(
              "relative inline-flex size-2 rounded-full",
              state === "live" && "bg-good",
              state === "offline" && "bg-critical",
              state === "checking" && "bg-ink-3",
            )}
          />
        </span>
        {state === "live" ? "Live" : state === "offline" ? "Offline" : "Connecting"}
        {data && (
          <span className="hidden font-mono text-[11px] text-ink-3 md:inline">
            {data.model_version}
          </span>
        )}
      </span>
    </Tip>
  );
}

function ThemeToggle() {
  const { theme, toggle } = useTheme();
  return (
    <Tip content={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}>
      <button
        type="button"
        onClick={toggle}
        aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
        className="inline-flex size-8 items-center justify-center rounded-lg text-ink-2 ring-1 ring-inset ring-line transition-colors hover:bg-surface-3 hover:text-ink"
      >
        {theme === "dark" ? <Sun size={15} /> : <Moon size={15} />}
      </button>
    </Tip>
  );
}

export function CommandPalette({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const navigate = useNavigate();
  const { toggle } = useTheme();
  const run = (fn: () => void) => {
    onOpenChange(false);
    fn();
  };
  const item =
    "flex cursor-pointer items-center gap-3 rounded-lg px-3 py-2 text-[13px] text-ink-2 data-[selected=true]:bg-surface-3 data-[selected=true]:text-ink";
  return (
    <Command.Dialog
      open={open}
      onOpenChange={onOpenChange}
      label="Command menu"
      overlayClassName="fixed inset-0 z-40 bg-black/50 backdrop-blur-sm"
      contentClassName="fixed left-1/2 top-[18%] z-50 w-[min(560px,calc(100vw-32px))] -translate-x-1/2 overflow-hidden rounded-2xl border border-line-strong bg-surface shadow-2xl"
    >
      <div className="flex items-center gap-2 border-b border-line px-4">
        <Search size={15} className="text-ink-3" aria-hidden />
        <Command.Input
          placeholder="Jump to a view, preset or action…"
          className="h-12 w-full bg-transparent text-[14px] text-ink outline-none placeholder:text-ink-3"
        />
        <Kbd>Esc</Kbd>
      </div>
      <Command.List className="max-h-80 overflow-y-auto p-2">
        <Command.Empty className="px-3 py-6 text-center text-[13px] text-ink-3">
          No matches.
        </Command.Empty>
        <Command.Group heading="Views" className="[&_[cmdk-group-heading]]:eyebrow [&_[cmdk-group-heading]]:px-3 [&_[cmdk-group-heading]]:py-2">
          {NAV.map((n) => (
            <Command.Item key={n.to} className={item} onSelect={() => run(() => navigate(n.to))}>
              <n.icon size={15} aria-hidden /> {n.label}
              <span className="ml-auto text-[11px] text-ink-3">{n.hint}</span>
            </Command.Item>
          ))}
        </Command.Group>
        <Command.Group heading="Actions" className="[&_[cmdk-group-heading]]:eyebrow [&_[cmdk-group-heading]]:px-3 [&_[cmdk-group-heading]]:py-2">
          <Command.Item
            className={item}
            onSelect={() => run(() => navigate("/playground?preset=suspicious"))}
          >
            <FlaskConical size={15} aria-hidden /> Score the suspicious example
          </Command.Item>
          <Command.Item
            className={item}
            onSelect={() => run(() => navigate("/playground?preset=typical"))}
          >
            <FlaskConical size={15} aria-hidden /> Score a typical transaction
          </Command.Item>
          <Command.Item className={item} onSelect={() => run(() => navigate("/threshold"))}>
            <Gauge size={15} aria-hidden /> Find the cheapest threshold
          </Command.Item>
          <Command.Item className={item} onSelect={() => run(toggle)}>
            <Sun size={15} aria-hidden /> Toggle theme
          </Command.Item>
        </Command.Group>
      </Command.List>
    </Command.Dialog>
  );
}

export function AppShell({ children }: { children: ReactNode }) {
  const [paletteOpen, setPaletteOpen] = useState(false);
  const location = useLocation();
  const current = NAV.find((n) => n.to === location.pathname) ?? NAV[0];

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setPaletteOpen((o) => !o);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => {
    document.title = `${current.label} · FraudLens`;
  }, [current.label]);

  return (
    <div className="mx-auto flex min-h-dvh max-w-[1560px]">
      <a
        href="#main"
        className="sr-only z-50 rounded-lg bg-accent px-3 py-2 text-white focus:not-sr-only focus:fixed focus:left-4 focus:top-4"
      >
        Skip to content
      </a>
      <aside className="sticky top-0 hidden h-dvh w-64 shrink-0 flex-col border-r border-line px-3 py-5 lg:flex">
        <Brand />
        <nav className="mt-8 flex flex-col gap-0.5" aria-label="Views">
          {NAV.map((n) => (
            <NavLink
              key={n.to}
              to={n.to}
              end
              className={({ isActive }) =>
                clsx(
                  "relative flex items-center gap-3 rounded-lg px-3 py-2 text-[13px] font-medium transition-colors",
                  isActive ? "text-ink" : "text-ink-3 hover:bg-surface-2 hover:text-ink",
                )
              }
            >
              {({ isActive }) => (
                <>
                  {isActive && (
                    <motion.span
                      layoutId="nav-active"
                      className="absolute inset-0 rounded-lg bg-surface-3 ring-1 ring-inset ring-line"
                      transition={{ type: "spring", stiffness: 500, damping: 38 }}
                    />
                  )}
                  <n.icon size={16} className="relative" aria-hidden />
                  <span className="relative">{n.label}</span>
                </>
              )}
            </NavLink>
          ))}
        </nav>
        <button
          type="button"
          onClick={() => setPaletteOpen(true)}
          className="mt-6 flex items-center gap-2 rounded-lg px-3 py-2 text-[12px] text-ink-3 ring-1 ring-inset ring-line transition-colors hover:text-ink"
        >
          <CommandIcon size={13} aria-hidden /> Command menu
          <span className="ml-auto flex gap-1">
            <Kbd>Ctrl</Kbd>
            <Kbd>K</Kbd>
          </span>
        </button>
        <div className="mt-auto rounded-xl bg-surface-2 p-3 text-[11px] leading-relaxed text-ink-3 ring-1 ring-inset ring-line">
          Portfolio project on a public 2013 dataset (ULB, ODbL). <strong className="font-semibold text-ink-2">Not a production system.</strong>
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-30 border-b border-line bg-page/80 backdrop-blur-xl">
          <div className="flex h-14 items-center gap-3 px-4 sm:px-6 lg:px-8">
            <div className="lg:hidden">
              <Brand />
            </div>
            <div className="hidden min-w-0 lg:block">
              <p className="truncate text-[13px] text-ink-3">
                FraudLens <span className="mx-1.5 text-axis">/</span>
                <span className="text-ink">{current.label}</span>
              </p>
            </div>
            <div className="ml-auto flex items-center gap-2">
              <HealthDot />
              <Button
                variant="ghost"
                size="sm"
                className="lg:hidden"
                aria-label="Open command menu"
                onClick={() => setPaletteOpen(true)}
              >
                <CommandIcon size={15} />
              </Button>
              <ThemeToggle />
            </div>
          </div>
          <nav
            className="flex gap-1 overflow-x-auto px-4 pb-2 [scrollbar-width:none] sm:px-6 lg:hidden [&::-webkit-scrollbar]:hidden"
            aria-label="Views"
          >
            {NAV.map((n) => (
              <NavLink
                key={n.to}
                to={n.to}
                end
                className={({ isActive }) =>
                  clsx(
                    "flex shrink-0 items-center gap-1.5 rounded-full px-3 py-1.5 text-[12px] font-medium",
                    isActive ? "bg-surface-3 text-ink ring-1 ring-inset ring-line" : "text-ink-3",
                  )
                }
              >
                <n.icon size={13} aria-hidden /> {n.label}
              </NavLink>
            ))}
          </nav>
        </header>

        <main id="main" className="flex-1 px-4 py-6 sm:px-6 lg:px-8 lg:py-8">
          {children}
        </main>
        <footer className="border-t border-line px-4 py-5 text-[12px] text-ink-3 sm:px-6 lg:px-8">
          FraudLens · results describe one public dataset of September 2013 card transactions
          and are not evidence of production performance.
        </footer>
      </div>
      <CommandPalette open={paletteOpen} onOpenChange={setPaletteOpen} />
    </div>
  );
}

export function PageHeader({
  title,
  description,
  actions,
}: {
  title: string;
  description: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4 lg:mb-8">
      <div className="max-w-3xl">
        <h1 className="text-[26px] font-semibold tracking-[-0.025em] text-ink sm:text-[30px]">
          {title}
        </h1>
        <p className="mt-2 text-[14px] leading-relaxed text-ink-2">{description}</p>
      </div>
      {actions}
    </div>
  );
}

export function Reveal({ children, delay = 0 }: { children: ReactNode; delay?: number }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, delay, ease: [0.22, 1, 0.36, 1] }}
    >
      {children}
    </motion.div>
  );
}

export class ErrorBoundary extends Component<
  { children: ReactNode; resetKey?: string },
  { error: Error | null }
> {
  state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidUpdate(prev: { resetKey?: string }) {
    if (prev.resetKey !== this.props.resetKey && this.state.error) this.setState({ error: null });
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("[FraudLens] view crashed", error, info.componentStack);
  }

  render() {
    if (this.state.error) {
      return (
        <ErrorState
          error={this.state.error}
          onRetry={() => this.setState({ error: null })}
        />
      );
    }
    return this.props.children;
  }
}

