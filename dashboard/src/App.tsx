import { Navigate, Route, Routes, useLocation } from "react-router";
import { AppShell, ErrorBoundary } from "./components/layout";
import { ModelService } from "./views/ModelService";
import { Overview } from "./views/Overview";
import { Playground } from "./views/Playground";
import { ThresholdExplorer } from "./views/ThresholdExplorer";

export function App() {
  const location = useLocation();
  return (
    <AppShell>
      <ErrorBoundary resetKey={location.pathname}>
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/threshold" element={<ThresholdExplorer />} />
          <Route path="/playground" element={<Playground />} />
          <Route path="/model" element={<ModelService />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </ErrorBoundary>
    </AppShell>
  );
}
