// Runs a .ttpu.json program on the real tiny-tpu RTL (WASM), in the browser, through the same
// executor the Node tests use (public/runtime/ttpu-exec.mjs).
import type { TtpuProgram } from "./types";
import type { CycleState } from "./state-schema";

/** One chip pass: its RTL cycle count and, for the grid view, the state of every cycle. */
export interface PassTrace {
  id: number;
  cycles: number;
  states: CycleState[];
}

export interface TpuRun {
  outputs: Record<string, { shape: number[]; data: number[] }>;
  totalCycles: number;
  passes: PassTrace[];
  hostOps: Record<string, number>;
}

interface Executor {
  executeProgram: (
    p: unknown,
    i: unknown,
    options: { states: boolean },
  ) => {
    outputs: TpuRun["outputs"];
    passes: PassTrace[];
    totalCycles: number;
    hostOps: Record<string, number>;
  };
}

let executor: Promise<Executor> | null = null;

function loadExecutor(): Promise<Executor> {
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
  const out = tpu.executeProgram(program, inputs, { states: true });
  return { outputs: out.outputs, totalCycles: out.totalCycles, passes: out.passes, hostOps: out.hostOps };
}
