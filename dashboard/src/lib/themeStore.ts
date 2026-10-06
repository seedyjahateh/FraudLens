import { storage } from "./storage";

export type Theme = "dark" | "light";
const KEY = "fraudlens.theme";

/** Dark by default; a saved choice wins. */
export function initialTheme(): Theme {
  const saved = storage.get<Theme | null>(KEY, null);
  return saved === "light" || saved === "dark" ? saved : "dark";
}

export const storeTheme = (theme: Theme) => storage.set(KEY, theme);
