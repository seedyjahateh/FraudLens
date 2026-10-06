import * as RadixTooltip from "@radix-ui/react-tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router";
import { Toaster } from "sonner";
import { App } from "./App";
import "./index.css";
import { ThemeProvider } from "./lib/theme";
import { initialTheme } from "./lib/themeStore";

document.documentElement.dataset.theme = initialTheme();

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retryDelay: (attempt) => Math.min(1000 * 2 ** attempt, 8000),
    },
  },
});

const root = document.getElementById("root");
if (!root) throw new Error("missing #root");

createRoot(root).render(
  <StrictMode>
    <ThemeProvider>
      <QueryClientProvider client={queryClient}>
        <RadixTooltip.Provider>
          <BrowserRouter basename="/dashboard">
            <App />
          </BrowserRouter>
          <Toaster
            position="bottom-right"
            toastOptions={{
              className:
                "!bg-surface !text-ink !border !border-line-strong !shadow-xl !rounded-xl !text-[13px]",
            }}
          />
        </RadixTooltip.Provider>
      </QueryClientProvider>
    </ThemeProvider>
  </StrictMode>,
);
