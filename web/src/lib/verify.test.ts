import { describe, expect, it } from "vitest";
import { flat, tpuExact } from "./verify";
import type { TpuResult } from "./types";

const tpu = (outputs: string[]) => ({ ok: true, passes: [], program: { outputs } }) as unknown as TpuResult;

describe("flat", () => {
  it("flattens nested oracle rows row-major", () => {
    expect(flat([[1, 2], [3, 4]])).toEqual([1, 2, 3, 4]);
    expect(flat(5)).toEqual([5]);
  });
});

describe("tpuExact", () => {
  const oracle = [[[1, 2], [3, 4]]];

  it("accepts a bit-exact match", () => {
    const run = { outputs: { C: { shape: [2, 2], data: [1, 2, 3, 4] } } } as never;
    expect(tpuExact(run, tpu(["C"]), oracle)).toBe(true);
  });

  it("rejects one wrong element", () => {
    const run = { outputs: { C: { shape: [2, 2], data: [1, 2, 3, 5] } } } as never;
    expect(tpuExact(run, tpu(["C"]), oracle)).toBe(false);
  });

  it("rejects a missing output or a length mismatch", () => {
    expect(tpuExact({ outputs: {} } as never, tpu(["C"]), oracle)).toBe(false);
    const short = { outputs: { C: { shape: [1, 2], data: [1, 2] } } } as never;
    expect(tpuExact(short, tpu(["C"]), oracle)).toBe(false);
  });
});
