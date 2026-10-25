// Pure result checks, kept out of the UI so they can be unit-tested.
import type { TpuRun } from "./tpu";
import type { Tensor, TpuResult } from "./types";

/** Flatten a nested oracle row (e.g. [[1, 2], [3, 4]]) into one list, row-major. */
export function flat(x: Tensor): number[] {
  return Array.isArray(x) ? x.flatMap(flat) : [x];
}

/** Whether every tiny-tpu output equals the oracle, compared by position (the program's output order). */
export function tpuExact(run: TpuRun, tpu: TpuResult, oracle: Tensor[]): boolean {
  const names = tpu.program?.outputs ?? [];
  return names.every((name, i) => {
    const got = run.outputs[name]?.data ?? [];
    const want = flat(oracle[i]);
    return got.length === want.length && got.every((v, k) => v === want[k]);
  });
}
