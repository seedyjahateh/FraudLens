import type { ReactNode } from "react";
import { useBundle } from "../lib/queries";
import type { Bundle } from "../lib/schemas";
import { ErrorState, Skeleton } from "./ui";

export function PageSkeleton() {
  return (
    <div aria-busy="true" aria-label="Loading" className="space-y-6">
      <Skeleton className="h-9 w-72" />
      <Skeleton className="h-4 w-[min(560px,90%)]" />
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        {Array.from({ length: 4 }, (_, i) => (
          <Skeleton key={i} className="h-[118px]" />
        ))}
      </div>
      <div className="grid gap-4 xl:grid-cols-3">
        <Skeleton className="h-[420px] xl:col-span-2" />
        <Skeleton className="h-[420px]" />
      </div>
    </div>
  );
}

/** Loads the evaluation bundle once; every view renders the same honest states. */
export function WithBundle({ children }: { children: (bundle: Bundle) => ReactNode }) {
  const query = useBundle();
  if (query.isPending) return <PageSkeleton />;
  if (query.isError) return <ErrorState error={query.error} onRetry={() => void query.refetch()} />;
  return <>{children(query.data)}</>;
}
