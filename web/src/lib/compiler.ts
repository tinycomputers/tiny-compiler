// Main-thread client for the Pyodide worker (public/compiler-worker.js).
import type { Analysis } from "./types";

let worker: Worker | null = null;
let nextId = 1;
const pending = new Map<number, { resolve: (a: Analysis) => void; reject: (e: Error) => void }>();

function getWorker(): Worker {
  if (!worker) {
    worker = new Worker("/compiler-worker.js");
    worker.onmessage = (event: MessageEvent) => {
      const { id, ok, result, error } = event.data;
      const p = pending.get(id);
      if (!p) return;
      pending.delete(id);
      if (ok) p.resolve(result as Analysis);
      else p.reject(new Error(error as string));
    };
    worker.onerror = (e) => {
      for (const p of pending.values()) p.reject(new Error(e.message || "worker failed"));
      pending.clear();
    };
  }
  return worker;
}

export function analyze(source: string, opt: number, seed: number): Promise<Analysis> {
  const id = nextId++;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    getWorker().postMessage({ id, source, opt, seed });
  });
}
