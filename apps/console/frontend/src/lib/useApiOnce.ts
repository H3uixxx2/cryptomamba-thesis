import { useEffect, useState } from "react";

/* Fetch once per page load and share the result between components.
 * Used for data that is read-only and not part of the console-wide store. */
export function useApiOnce<T>(load: () => Promise<T>, key: string) {
  const [state, setState] = useState<{ data: T | null; error: string | null; loading: boolean }>(
    () => ({ data: cache.get(key) as T | null ?? null, error: null, loading: !cache.has(key) }),
  );

  useEffect(() => {
    if (cache.has(key)) return;
    let alive = true;
    let pending = inflight.get(key) as Promise<T> | undefined;
    if (!pending) {
      pending = load();
      inflight.set(key, pending);
    }
    pending
      .then((data) => {
        cache.set(key, data);
        if (alive) setState({ data, error: null, loading: false });
      })
      .catch((error: unknown) => {
        inflight.delete(key);
        if (alive) setState({ data: null, error: error instanceof Error ? error.message : String(error), loading: false });
      });
    return () => {
      alive = false;
    };
  }, [key, load]);

  return state;
}

const cache = new Map<string, unknown>();
const inflight = new Map<string, Promise<unknown>>();
