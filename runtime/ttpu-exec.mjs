// ttpu-exec.mjs - the ONE executor for tiny-tpu programs (Node in tests, browser in the UI).
//
// A `.ttpu.json` program (docs/TTPU_FORMAT.md) is a list of commands. Only `pass` runs on the
// chip (the real tiny-tpu RTL, Verilator -> WASM); every other command is host work done here
// in plain integers and counted separately, so nothing host-side is passed off as chip work.

export const TILE = 4;
export const TILE_ELEMS = TILE * TILE;

const I32_MIN = -(2 ** 31);
const I32_MAX = 2 ** 31 - 1;
const DTYPE_RANGE = { i8: [-128, 127], i32: [I32_MIN, I32_MAX] };

const size = (shape) => shape.reduce((n, d) => n * d, 1);

function fits(v, what) {
  if (v < I32_MIN || v > I32_MAX) throw new RangeError(`${what} does not fit in i32`);
  return v;
}

/** Flatten nested lists into a row-major array, checking the shape and the dtype range. */
function flatten(nested, shape, dtype, what) {
  const out = [];
  const walk = (x, depth) => {
    if (depth === shape.length) {
      if (!Number.isInteger(x)) throw new Error(`${what}: expected an integer`);
      out.push(x);
      return;
    }
    if (!Array.isArray(x) || x.length !== shape[depth]) {
      throw new Error(`${what}: expected shape [${shape}]`);
    }
    for (const item of x) walk(item, depth + 1);
  };
  walk(nested, 0);
  const [lo, hi] = DTYPE_RANGE[dtype];
  if (out.some((v) => v < lo || v > hi)) throw new Error(`${what}: values outside ${dtype}`);
  return out;
}

/** Zero-padded 4x4 window of a rank-2 buffer at (r0, c0), as a flat row-major array of 16. */
function getTile(buf, r0, c0) {
  const [rows, cols] = buf.shape;
  const tile = new Array(TILE_ELEMS).fill(0);
  for (let i = 0; i < TILE; i++) {
    for (let j = 0; j < TILE; j++) {
      if (r0 + i < rows && c0 + j < cols) tile[i * TILE + j] = buf.data[(r0 + i) * cols + c0 + j];
    }
  }
  return tile;
}

/** Apply bias then relu to the in-bounds part of dst's window at (r0, c0). */
function epilogue(dst, r0, c0, cmd, buffers) {
  const [rows, cols] = dst.shape;
  const bias = cmd.bias === undefined ? null : buffers.get(cmd.bias).data;
  for (let i = 0; i < TILE && r0 + i < rows; i++) {
    for (let j = 0; j < TILE && c0 + j < cols; j++) {
      const k = (r0 + i) * cols + c0 + j;
      if (bias) dst.data[k] = fits(dst.data[k] + bias[c0 + j], "bias add");
      if (cmd.relu) dst.data[k] = Math.max(dst.data[k], 0);
    }
  }
}

/**
 * @param {() => Promise<any>} factory  the `createTinyTpu` default export of tiny_tpu.mjs
 */
export async function createTpu(factory) {
  const mod = await factory();

  /** Run one 4x4 pass on the RTL. Returns { c: number[16], cycles, states? }. */
  function runPass(a, b, { states = false } = {}) {
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
  }

  /**
   * Execute a `.ttpu.json` program.
   * @returns {{outputs: Object, passes: {id:number, cycles:number}[], totalCycles:number,
   *            hostOps: Object}}
   */
  function executeProgram(program, inputs, { states = false } = {}) {
    if (program.format !== "ttpu" || program.version !== 1) {
      throw new Error(`unsupported program format ${program.format} v${program.version}`);
    }
    const buffers = new Map();
    for (const [name, b] of Object.entries(program.buffers)) {
      const shape = b.shape;
      let data;
      if (b.kind === "const") data = flatten(b.data, shape, b.dtype, `const '${name}'`);
      else if (b.kind === "input") {
        if (!(name in inputs)) throw new Error(`missing input '${name}'`);
        data = flatten(inputs[name], shape, b.dtype, `input '${name}'`);
      } else data = new Array(size(shape)).fill(0);
      buffers.set(name, { shape, dtype: b.dtype, data });
    }

    const tmp = new Map();
    const passes = [];
    const hostOps = { alloc: 0, acc: 0, epilogue: 0, host_add: 0, host_relu: 0 };
    for (const cmd of program.commands) {
      if (cmd.op === "pass") {
        const { c, cycles, states: st } = runPass(
          getTile(buffers.get(cmd.a.buf), cmd.a.r, cmd.a.c),
          getTile(buffers.get(cmd.b.buf), cmd.b.r, cmd.b.c),
          { states },
        );
        tmp.set(cmd.id, c);
        passes.push(states ? { id: cmd.id, cycles, states: st } : { id: cmd.id, cycles });
        continue;
      }
      hostOps[cmd.op] += 1;
      if (cmd.op === "alloc") {
        buffers.get(cmd.buf).data.fill(0);
      } else if (cmd.op === "acc") {
        const dst = buffers.get(cmd.dst.buf);
        const [rows, cols] = dst.shape;
        const t = tmp.get(cmd.pass);
        for (let i = 0; i < TILE && cmd.dst.r + i < rows; i++) {
          for (let j = 0; j < TILE && cmd.dst.c + j < cols; j++) {
            const k = (cmd.dst.r + i) * cols + cmd.dst.c + j;
            dst.data[k] = fits(dst.data[k] + t[i * TILE + j], "accumulate");
          }
        }
        epilogue(dst, cmd.dst.r, cmd.dst.c, cmd, buffers);
      } else if (cmd.op === "epilogue") {
        epilogue(buffers.get(cmd.dst.buf), cmd.dst.r, cmd.dst.c, cmd, buffers);
      } else if (cmd.op === "host_add") {
        const dst = buffers.get(cmd.dst);
        const a = buffers.get(cmd.a);
        const b = buffers.get(cmd.b);
        // same shape, or a rank-1 operand broadcast over the rows of a rank-2 one
        const cols = dst.shape[dst.shape.length - 1];
        const at = (x, k) => (x.shape.length === dst.shape.length ? x.data[k] : x.data[k % cols]);
        for (let k = 0; k < dst.data.length; k++) dst.data[k] = fits(at(a, k) + at(b, k), "add");
      } else if (cmd.op === "host_relu") {
        const dst = buffers.get(cmd.dst);
        const src = buffers.get(cmd.src);
        for (let k = 0; k < dst.data.length; k++) dst.data[k] = Math.max(src.data[k], 0);
      } else {
        throw new Error(`unknown command '${cmd.op}'`);
      }
    }

    const outputs = {};
    for (const name of program.outputs) {
      const { shape, data } = buffers.get(name);
      outputs[name] = { shape, data };
    }
    return {
      outputs,
      passes,
      totalCycles: passes.reduce((n, p) => n + p.cycles, 0),
      hostOps,
    };
  }

  return { runPass, executeProgram };
}
