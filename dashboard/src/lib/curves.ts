export const RECALL_GRID = Array.from({ length: 201 }, (_, i) => i / 200);

/**
 * Interpolated precision at each recall level: the best precision reachable at that recall
 * or higher. Resampling every model onto one recall grid lets one crosshair read all series.
 */
export function interpolatedPrecision(
  recall: number[],
  precision: number[],
  grid: number[] = RECALL_GRID,
): number[] {
  const pts = recall.map((r, i) => [r, precision[i] ?? 0] as const).sort((a, b) => a[0] - b[0]);
  const out: number[] = new Array<number>(grid.length).fill(0);
  let j = pts.length - 1;
  let best = 0;
  for (let g = grid.length - 1; g >= 0; g--) {
    const r = grid[g] ?? 0;
    while (j >= 0 && (pts[j]?.[0] ?? -1) >= r - 1e-12) {
      best = Math.max(best, pts[j]?.[1] ?? 0);
      j--;
    }
    out[g] = best;
  }
  return out;
}
