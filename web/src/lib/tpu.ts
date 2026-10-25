// Runs a .ttpu.json program on the real tiny-tpu RTL (WASM), in the browser, through the same
// executor the Node tests use (public/runtime/ttpu-exec.mjs).
import type { TtpuProgram } from "./types";

export interface TpuRun {
  outputs: Record<string, { shape: number[]; data: number[] }>;
  totalCycles: number;
  passes: number;
  hostOps: Record<string, number>;
}

let executor: Promise<{ executeProgram: (p: unknown, i: unknown) => unknown }> | null = null;


function loadExecutor() {
  if (!executor) {
    executor = (async () => {
      // Public files are served as-is; a variable URL keeps Vite from trying to bundle them.
      const wasmUrl: string = "/tiny_tpu.mjs";
      const execUrl: string = "/runtime/ttpu-exec.mjs";
      const { default: factory } = await import(/* @vite-ignore */ wasmUrl);
      const { createTpu } = await import(/* @vite-ignore */ execUrl);
      return createTpu(factory);
    })();
  }
  return executor;
}

export async function runTpu(program: TtpuProgram, inputs: Record<string, number[]>): Promise<TpuRun> {
  const tpu = await loadExecutor();
  const out = tpu.executeProgram(program, inputs) as {
    outputs: TpuRun["outputs"];
    passes: unknown[];
    totalCycles: number;
    hostOps: Record<string, number>;
  };
  return { outputs: out.outputs, totalCycles: out.totalCycles, passes: out.passes.length, hostOps: out.hostOps };
}
