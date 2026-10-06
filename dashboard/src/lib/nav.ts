import { Activity, FlaskConical, LayoutDashboard, SlidersHorizontal } from "lucide-react";

export const NAV = [
  { to: "/", label: "Overview", icon: LayoutDashboard, hint: "Results on the held-out test slice" },
  { to: "/threshold", label: "Threshold & cost", icon: SlidersHorizontal, hint: "Explore the cost trade-off" },
  { to: "/playground", label: "Scoring playground", icon: FlaskConical, hint: "Score a transaction live" },
  { to: "/model", label: "Model & service", icon: Activity, hint: "Provenance, health, latency" },
] as const;

export const AXIS_TICK = { fontSize: 11 };
