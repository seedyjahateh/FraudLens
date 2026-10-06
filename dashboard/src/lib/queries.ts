import { useMutation, useQuery } from "@tanstack/react-query";
import { api, ApiError } from "./api";
import type { Transaction } from "./schemas";

const retry = (count: number, error: unknown) =>
  // Contract errors and 4xx will not fix themselves; network blips and 5xx might.
  !(error instanceof ApiError && (error.kind === "schema" || (error.status >= 400 && error.status < 500))) &&
  count < 3;

export const useBundle = () =>
  useQuery({ queryKey: ["bundle"], queryFn: api.bundle, staleTime: Infinity, retry });

export const useModel = () =>
  useQuery({ queryKey: ["model"], queryFn: api.model, staleTime: 60_000, retry });

export const useHealth = () =>
  useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    refetchInterval: 10_000,
    retry: false,
  });

export const useStats = () =>
  useQuery({ queryKey: ["stats"], queryFn: api.stats, refetchInterval: 5_000, retry });

export const usePresets = () =>
  useQuery({ queryKey: ["presets"], queryFn: api.presets, staleTime: Infinity, retry });

export const useScore = () =>
  useMutation({ mutationFn: (transaction: Transaction) => api.score(transaction) });
