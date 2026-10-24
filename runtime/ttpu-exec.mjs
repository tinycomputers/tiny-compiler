// ttpu-exec.mjs - the ONE executor for tiny-tpu passes (Node in tests, browser in the UI).
//
// Phase 0 scope: drive single 4x4 passes on the real tiny-tpu RTL (Verilator -> WASM).
// Phase 3 grows this into the full `.ttpu.json` command list (pass / acc / relu / ...).
//
// Inputs are flat row-major arrays of 16 numbers (int8 for a/b); results are int32[16].

export const TILE = 4;
export const TILE_ELEMS = TILE * TILE;

/**
 * @param {() => Promise<any>} factory  the `createTinyTpu` default export of tiny_tpu.mjs
 */
export async function createTpu(factory) {
  const mod = await factory();
  return {
    /** Run one 4x4 pass. Returns { c: number[16], cycles: number, states?: CycleState[] }. */
    runPass(a, b, { states = false } = {}) {
      if (a.length !== TILE_ELEMS || b.length !== TILE_ELEMS) {
        throw new Error(`a and b must each have ${TILE_ELEMS} elements`);
      }
      const sim = new mod.TinyTpuSim();
      try {
        sim.reset();
        sim.loadA(a);
        sim.loadB(b);
        sim.start();
        const raw = sim.run();
        const out = { c: Array.from(sim.getResult(), Number), cycles: sim.getCycleCount() };
        if (states) out.states = Array.from(raw);
        return out;
      } finally {
        sim.delete();
      }
    },
  };
}
