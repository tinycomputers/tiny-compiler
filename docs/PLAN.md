# TinyCompiler — End-to-End Build Plan

> **One tiny tensor language, two real chips.**
> Write `C = relu(A @ B)` once; TinyCompiler lowers it either to **tiny-gpu** SIMT assembly
> (`.tgpu`) or to a **tiny-tpu** tile schedule (`.ttpu.json`), runs it on the real simulator /
> real RTL, and shows every pass in the browser.
> Fourth project in the series after [tiny-gpu](https://github.com/deaneeth/tiny-gpu),
> [tiny-tpu](https://github.com/deaneeth/tiny-tpu) and tiny-cpu. It is the software layer that
> ties the hardware projects together.

Same rule as the hardware projects: **nothing on screen is faked.** Every number in the UI
comes from the compiler's real IR or from running the emitted code on the unmodified
tiny-gpu simulator or the compiled tiny-tpu RTL (WASM). Host-side work (tile accumulation,
padding) is labelled as host work, never drawn as if the chip did it.

---

## 1. Goals and non-goals

### Goals
1. **A real multi-level compiler, small enough to read in an afternoon.**
   Source → AST → Graph IR → target IR (Tile IR for TPU, Kernel IR for GPU) → machine output.
   A pass manager, an IR verifier, and a printer for every level (`--print-after-all`).
2. **Two backends, one program.** The same `.tiny` file compiles to both chips. The output is
   bit-exact with a numpy reference on both.
3. **Verified for real.** Interpreters for every IR level (translation validation after every
   pass), differential fuzzing (numpy vs GPU vs TPU), golden snapshots of emitted code, and
   pinned cycle counts.
4. **Optimizations you can see.** Fusion, loop unrolling, strength reduction and shared-memory
   tiling on the GPU, plus tiling and padding decisions on the TPU. Each one has a measured
   before/after on the real target.
5. **Compiler Explorer in the browser.** A Godbolt-style page: edit source, step through the
   passes, pick a target, run it live, and compare the two chips head to head.

### Non-goals (v1)
- No floats. Integers only: `i8` and `i32`, as the hardware supports.
- No general ML framework: a fixed op set (§3.2), static shapes only, no control flow in the
  language.
- No changes to tiny-gpu or tiny-tpu in v1. **Target the chips as they are.** ISA gaps get
  written down and become stretch goals (§13) with upstream PRs.
- No requantization (`i32 → i8`) in v1, so chained TPU matmuls are rejected (§3.3).
- No autotuner in v1 (stretch).

---

## 2. What the targets actually provide (from reading their source)

This section sets the constraints for everything else. Re-check it if either submodule is bumped.

### 2.1 tiny-gpu (Python simulator, `src/tinygpu/`)

| Property | Fact | Consequence for the compiler |
|---|---|---|
| Execution model | Per-thread PC (each thread branches on its own); `grid=(num_blocks, threads_per_block)` | Map output `C[i][j]` → block `i`, thread `j`. Divergent `relu` branches are legal and cheap. |
| Registers | `num_registers` is a constructor arg; **R5 = block_id, R6 = thread_in_block, R7 = global tid** (set by the simulator); kernel args go into R0..Rk | Allocatable pool = all registers except R5–R7. Bake base addresses in as immediates; don't use kernel args. |
| ISA | `SET ADD MUL LD ST JMP BEQ BNE CMP BRGT BRLT BRZ SYNC SYNCB SHLD SHST CSWAP`; operands may be registers or immediates | **No SUB, DIV, shifts or MAX.** Loops use `CMP` + `BRLT`; `relu` = `CMP r,0 / BRGT skip / SET r,0`. |
| Memory | Global `memory`: int32 words, word-addressed, size `mem_size`; per-block `shared[block][shared_size]` | The memory planner lays out word addresses. `i8` values are stored one per int32 word. |
| Arithmetic | Results are written into an `np.int32` array; a Python int outside int32 **raises** instead of wrapping | Range analysis must prove there is no i32 overflow, or reject the program (§5.4). |
| Cycle semantics | `step()` runs **consecutive non-control instructions in one cycle**; a cycle ends at a branch/jump/sync | "Cycles" mostly count branches. Always report **instructions issued** as well as cycles, and say so in the UI (§11 risk). |
| Grid resize quirk | `set_grid` resizing the thread count leaves R7 = 0 for threads it adds | The runner builds `TinyGPU(num_threads=blocks*tpb, ...)` up front. File an upstream fix. |
| Memory between launches | `load_kernel` resets PCs and registers but **not** global memory | Multi-kernel programs share one global memory, so intermediate buffers persist. |

### 2.2 tiny-tpu (SystemVerilog → Verilator → WASM, `rtl/` + `wasm/`)

| Property | Fact | Consequence for the compiler |
|---|---|---|
| Compute | One **4×4 weight-stationary** pass: `C = A·B`, **int8 × int8 → int32** | All TPU matmuls are tiled to 4×4 and zero-padded. Operands must be `i8`. |
| Pass cost | `LOAD_WEIGHTS 4 + STREAM 7 + DRAIN 3 = 14` cycles from `start` to `done`, plus 1 cycle for `C[3][3]` to latch. **`getCycleCount()` reports 15** (measured in Phase 0) | Cost model: `passes × 15` as measured, `× 14` start→done. Pinned in `test_targets_contract.py`. |
| State between passes | Every pass starts from psum = 0 and always reloads the weights | Accumulation over K-tiles is **host work**. Loop order cannot save TPU cycles today (this motivates the co-design stretch goal in §13). |
| No epilogue | No bias, relu or accumulate in hardware | `add` and `relu` after a TPU matmul are host ops in the schedule. |
| WASM API | `TinyTpuSim: reset, loadA, loadB, start, step, run → CycleState[], getResult → int32[16], getCycleCount` | The `.ttpu` executor drives exactly this API, in Node (tests) and in the browser (UI). |

---

## 3. The language (`.tiny`)

### 3.1 Example
```
# examples/mm_relu.tiny
input  A : i8[4, 8]
input  B : i8[8, 4]
C = relu(A @ B)          # inferred i32[4, 4]
output C
```

### 3.2 v1 grammar and ops
```
program   := stmt*
stmt      := "input" NAME ":" type
           | "const" NAME ":" type "=" literal
           | NAME "=" expr
           | "output" NAME ("," NAME)*
type      := ("i8" | "i32") "[" INT ("," INT)* "]"        # rank 1 or 2
expr      := expr "@" expr | expr "+" expr | "relu" "(" expr ")" | NAME | "(" expr ")"
```
| Op | Typing rule | Notes |
|---|---|---|
| `a @ b` | `i8[M,K] @ i8[K,N] → i32[M,N]`; on the GPU also `i32 @ i32 → i32` | The only op that changes shape |
| `a + b` | same shape, or `[M,N] + [N]` (row broadcast for bias) | `i8` is promoted to `i32` |
| `relu(a)` | shape and type preserved | |

Diagnostics carry `line:col` spans and caret snippets (`frontend/diagnostics.py`). The error
messages are a feature: Vitest/pytest snapshot them.

### 3.3 Target legality (checked by `passes/legalize_<target>.py`)
- **TPU:** both matmul operands must be `i8` graph inputs or constants. So `relu(A@B) @ D` is
  rejected with: "tiny-tpu needs i8 operands; this value is i32 (requant is v2)".
- **GPU:** total buffer words ≤ the `mem_size` budget; threads per launch ≤ 1024 (a simulator
  speed cap).
- **Both:** range analysis proves every intermediate fits in i32.

---

## 4. Compiler architecture

```
 .tiny ──▶ lexer/parser ──▶ AST ──▶ Graph IR (TG) ──▶ graph passes ─┬─▶ TPU: Tile IR ──▶ tile passes ──▶ .ttpu.json
                                     (SSA, typed,     shape-infer   │
                                      loc-tagged)     canonicalize  └─▶ GPU: Kernel IR ──▶ kernel passes ──▶ regalloc ──▶ .tgpu + launch.json
                                                      cse/dce
                                                      const-fold
                                                      fuse
                                                      range-check
                                                      legalize
 every level: printer · verifier · interpreter            every pass: --print-after · --verify-each (interp == numpy)
```

### 4.1 Graph IR (target-independent)
```
func @main(%A: i8[4,8], %B: i8[8,4]) -> (i32[4,4]) {
  %0 = matmul %A, %B : i32[4,4]        loc("mm_relu.tiny":4:10)
  %1 = relu %0       : i32[4,4]        loc("mm_relu.tiny":4:5)
  return %1
}
// after fuse:
  %0 = matmul{epilogue=relu} %A, %B : i32[4,4]
```
SSA values, explicit types, and a `loc` on every op. The locs survive every lowering, which is
what lets the UI highlight the source line for any line of assembly (§9).

### 4.2 TPU path: Tile IR → `.ttpu.json`
```
tpu.schedule @main tile=4x4 {
  host.alloc %C : i32[4,4] = 0
  tpu.pass #0 a=%A[0:4,0:4] b=%B[0:4,0:4] -> %p0      // 15 cyc
  host.acc   %C[0:4,0:4] += %p0
  tpu.pass #1 a=%A[0:4,4:8] b=%B[4:8,0:4] -> %p1      // 15 cyc
  host.acc   %C[0:4,0:4] += %p1
  host.relu  %C
}
```
- **Tiling pass:** `ceil(M/4)·ceil(N/4)·ceil(K/4)` passes, zero padding at the edges. The
  schedule order matches tiny-tpu's `tilingSchedule()` (output tile row-major, K innermost),
  so the existing tiny-tpu L3 view can display it.
- **Tile passes:** skip passes whose A or B tile is all zero (constants only), sink `relu`/bias
  into the last `host.acc` of each output tile, and report utilization (useful MACs ÷ 64 per pass: a pass is 4×4×4 MACs).
- **Emit:** `.ttpu.json` = a buffer table, int8 tile payloads (or slice refs), and a command list
  (`pass`, `acc`, `relu`, `add_bias`, `store`). The format is versioned and has a JSON Schema
  in `docs/TTPU_FORMAT.md`.

### 4.3 GPU path: Kernel IR → `.tgpu`
```
kernel @mm_relu grid=(4 blocks, 4 threads) {
  %i = block_id                     // R5
  %j = thread_in_block              // R6
  %acc = const 0
  for %k = 0 to 8 {
    %a = ld [A + %i*8 + %k]
    %b = ld [B + %k*4 + %j]
    %acc = add %acc, (mul %a, %b)
  }
  %acc = relu %acc
  st [C + %i*4 + %j], %acc
}
```
- **Lowering:** each Graph op becomes one kernel (one thread per output element). Fused ops
  become one kernel. A multi-op program becomes several launches over shared global memory.
- **Memory planner:** word addresses for inputs, intermediates and outputs, with liveness-based
  reuse of intermediate buffers; it emits `mem_size`.
- **Kernel passes (by opt level):**
  - `-O0`: naive loop.
  - `-O1`: loop-invariant code motion + strength reduction (hoist `%i*8` and step pointers by a
    constant instead of re-multiplying).
  - `-O2`: full unroll of the K loop when K ≤ 16, then shared-memory tiling: a block
    cooperatively loads row `i` of A into `shared` (`SHST`), `SYNCB`, and the inner loop reads it
    with `SHLD`. Global `LD`s drop from `2·M·N·K` to `M·K + M·N·K`.
- **Register allocation:** linear scan over virtual registers. Pool = `R0–R4, R8..R(n-1)`
  (R5–R7 reserved). The budget defaults to 16 (`--gpu-regs`). No spilling in v1: running out is
  a clear compile error naming the live values.
- **Emit:** `.tgpu` text (assembled by tiny-gpu's own `assemble_file`, so we never bypass its
  assembler), plus `launch.json`: `{kernels: [{file, grid, shared_size}], mem_size,
  buffers: {name: [addr, shape]}}`.
- Example of the emitted `relu`: `CMP R0, 0` / `BRGT L_pos` / `SET R0, 0` / `L_pos:`.

### 4.4 Cost models (`tpu/cost.py`, `gpu/cost.py`)
Static estimates printed by `tinyc --stats` and shown before you press Run:
- TPU: `passes × 15` RTL cycles, host op count, padding utilization.
- GPU: instructions issued, simulator cycles (as tiny-gpu defines them), global/shared loads
  and stores, registers used.
After running, measured numbers sit next to the estimates. **Estimate and measurement must
match exactly** for the TPU and for GPU instruction counts. That is a test (§5, layer 6).

---

## 5. Verification strategy

Six layers. Each phase in §10 says which layers must pass.

1. **Reference semantics:** `ref.py` evaluates the Graph IR with numpy int64, then asserts that
   every value fits in i32. This is the oracle.
2. **Translation validation:** there is an interpreter for the Graph IR, Tile IR and Kernel IR
   (Kernel IR is interpreted per thread, with virtual registers). With `--verify-each`, after
   **every** pass the current IR runs on seeded random inputs and must equal the oracle. A
   failure names the pass that broke it and prints IR before and after.
3. **Real targets:** the emitted `.tgpu` runs on the unmodified tiny-gpu simulator; `.ttpu.json`
   runs on the tiny-tpu WASM (Node). Results must be bit-exact with the oracle.
4. **Differential fuzzing (Hypothesis):** random legal programs (≤ 4 ops, dims 1–12, random
   int8 data, random consts) are compiled to both targets at every opt level and compared with
   the oracle. Hypothesis shrinking gives a minimal failing program, printed as `.tiny` source
   and saved to `tests/regressions/`. CI runs 200 examples with seed 42; nightly runs 5000.
5. **Golden snapshots:** the emitted `.tgpu`, `.ttpu.json` and diagnostics for every file in
   `examples/` are kept under `tests/golden/`. Regenerate them with `pytest --update-golden`;
   reviewers can read the diff.
6. **Perf pins:** `tests/perf_baseline.json` records TPU cycles and passes, plus GPU cycles,
   instructions and loads, per example × target × opt level. An unexpected change fails CI. A
   second check: the cost model equals the measurement.

Plus a **target contract test** (`tests/test_targets_contract.py`): after a submodule bump, the
tiny-gpu ISA table and the tiny-tpu WASM API still match §2. If not, it fails loudly before
anything else runs.

---

## 6. Runtime (`runtime/`)

| File | Role |
|---|---|
| `gpu_runner.py` | Builds `TinyGPU(num_threads=…, num_registers=…, mem_size=…)`, writes inputs, runs each launch from `launch.json` with `load_kernel` + `run_kernel`, reads outputs. A counting wrapper on the `INSTRUCTIONS` table records issued/LD/ST/SHLD/SHST counts (it only observes; semantics are unchanged). `history=False` mode for speed. |
| `ttpu-exec.mjs` | **The only `.ttpu` executor**, shared by Node and the browser. It drives `TinyTpuSim` per `pass` command and does host ops in plain JS integers; it returns results, per-pass `CycleState[]` (optional) and the cycle total. |
| `tpu_runner.py` | Calls `node runtime/ttpu-cli.mjs <prog.ttpu.json> <inputs.json>` and parses the JSON. Python never re-implements the executor. |

---

## 7. Repository layout (mirrors the sibling projects)

```
tiny-compiler/
├── AGENTS.md / CLAUDE.md / CONTEXT.md / README.md / LICENSE / .editorconfig
├── targets/                      # git submodules, pinned commits
│   ├── tiny-gpu/                 # imported as a package (src/tinygpu)
│   └── tiny-tpu/                 # web/public/tiny_tpu.{mjs,wasm} copied by scripts/sync-targets.sh
├── tinyc/
│   ├── frontend/  lexer.py parser.py ast.py diagnostics.py
│   ├── ir/        graph.py types.py printer.py verifier.py interp.py ref.py loc.py
│   ├── passes/    manager.py shape_infer.py canonicalize.py cse.py dce.py const_fold.py
│   │              fuse.py range_check.py legalize_tpu.py legalize_gpu.py
│   ├── tpu/       tile_ir.py lower.py tile_passes.py emit.py cost.py interp.py
│   ├── gpu/       kernel_ir.py lower.py memplan.py licm.py strength.py unroll.py
│   │              shared_tile.py regalloc.py emit.py cost.py interp.py
│   ├── driver.py  # pipelines per target/opt level
│   └── cli.py     # `tinyc file.tiny --target gpu|tpu|both -O2 --print-after-all --run --stats`
├── runtime/       gpu_runner.py tpu_runner.py ttpu-exec.mjs ttpu-cli.mjs
├── examples/      mm.tiny mm_relu.tiny mm_bias_relu.tiny mlp_gpu.tiny big_mm.tiny ...
├── tests/
│   ├── frontend/ ir/ passes/ tpu/ gpu/    # unit tests per module
│   ├── test_e2e.py            # examples × targets × opt levels vs oracle
│   ├── test_fuzz.py           # Hypothesis differential
│   ├── test_targets_contract.py
│   ├── golden/  regressions/  perf_baseline.json
├── docs/
│   ├── PLAN.md  ARCHITECTURE.md  LANGUAGE.md  IR.md  TTPU_FORMAT.md  VERIFICATION.md
│   └── MLIR_MAPPING.md        # how each IR maps to linalg / scf / affine (stretch, §13)
├── scripts/       env.sh sync-targets.sh build-wheel.sh
├── .github/workflows/  py.yml  fuzz-nightly.yml  web.yml
└── web/                       # Astro + React + Tailwind + shadcn (copy tiny-tpu config)
    ├── public/ tiny_tpu.{mjs,wasm}  pyodide/ (lazy)  tinyc-*.whl  tinygpu-*.whl
    └── src/
        ├── workers/compiler.worker.ts   # Pyodide: tinyc + tinygpu + numpy
        ├── lib/ ttpu-exec.ts (re-export runtime/ttpu-exec.mjs) wasm-loader.ts state-schema.ts ir-format.ts
        ├── hooks/ useCompiler.ts useTpuRun.ts useGpuRun.ts
        └── components/ (§9)
```

Python ≥ 3.12, 4-space indent, Black + Ruff, pytest, fixed seed 42. tiny-gpu is installed from
the submodule with `pip install -e targets/tiny-gpu --no-deps` plus numpy. Its matplotlib and
imageio dependencies are not needed, which also keeps the Pyodide bundle small.

---

## 8. Toolchain

| Tool | Used for |
|---|---|
| Python 3.12 + numpy + pytest + Hypothesis + Black + Ruff | compiler, tests, fuzzing |
| Node ≥ 20 | `.ttpu` executor in tests (loads tiny-tpu's `.mjs`) |
| Pyodide (pinned version) | running the compiler + tiny-gpu in the browser |
| pnpm + Astro + React + Vitest + Playwright | web (same as tiny-tpu) |

No Verilator or Emscripten is needed here: we use tiny-tpu's committed WASM artifact at the
pinned submodule commit. (Optional nightly job: rebuild it from the submodule and diff.)

---

## 9. Web app: Compiler Explorer (`web/`)

Same visual language as tiny-tpu and tiny-cpu, so the four projects look like one family.

```
┌───────────────────────────────────────────────────────────────────────────────┐
│ [Example ▾ mm_relu] [Target: GPU | TPU | Both] [-O0 -O1 -O2] [Compile] [Run ▶] │
├─────────────────────┬─────────────────────────────────────────────────────────┤
│ Source (.tiny)      │ Pass pipeline:  AST · Graph · fused · Tile/Kernel · asm │
│ CodeMirror + errors │ (one tab per pass; diff vs previous pass highlighted;   │
│ hover ⇄ IR / asm    │  hover any line → its source span lights up)            │
├─────────────────────┼─────────────────────────────────────────────────────────┤
│ Inputs A, B         │ Execution:  TPU PE grid live  |  GPU threads × cycles    │
│ (editable grids,    │ Results: C(numpy) C(gpu) C(tpu) ✓ bit-exact              │
│  randomize)         │ Stats: estimate vs measured (cycles, instrs, loads)     │
└─────────────────────┴─────────────────────────────────────────────────────────┘
```

| Level | Component | What it shows |
|---|---|---|
| **L1 Passes** | `PassExplorer.tsx` | Every IR after every pass, with diffs and source-location linking. You can switch passes off to see what each one does. |
| **L2 TPU schedule** | `TileTimeline.tsx` + `PEGrid.tsx` (copied from tiny-tpu) | A Gantt chart of `tpu.pass` and `host.*` commands; clicking a pass replays its real `CycleState[]` on the PE grid. Padding cells are hatched. |
| **L3 GPU kernel** | `KernelView.tsx` | The `.tgpu` listing with a per-thread PC marker, a register heatmap, a global/shared memory heatmap, and LD/ST counters. `-O0`/`-O2` side by side. |
| **L4 Head to head** | `Compare.tsx` | The same program on both chips: cycles, instructions, memory traffic, utilization, plus a size-sweep chart (4…16). The cycle-semantics caveat from §2.1 is shown inline. |

**How it runs:** Pyodide loads in a Web Worker (lazily, on the first Compile) with the `tinyc`
and `tinygpu` wheels. Compile returns all IR dumps plus the artifacts. GPU runs happen in the
same worker. TPU runs use `ttpu-exec` + the WASM on the main thread. Nothing the compiler does
is re-implemented in TypeScript.

Web tests: Vitest for `ttpu-exec` (fixtures shared with Python tests), the IR-format parser
and the location mapper; one Playwright smoke test (open `/app`, compile `mm_relu` for Both, Run,
assert "bit-exact" for both targets).

---

## 10. Phases, deliverables and exit criteria

Estimates assume one person working part-time, at the pace of the earlier projects.

### Phase 0: Scaffold + "hello targets" (1 week)
- Repo with the §7 layout, submodules pinned, `scripts/env.sh`, CI skeletons, CLAUDE/AGENTS/CONTEXT.
- `gpu_runner.py` runs tiny-gpu's own `examples/vector_add.tgpu`; `ttpu-exec.mjs` runs one
  hand-written 4×4 pass on the WASM. Both are checked against numpy in pytest.
- `test_targets_contract.py` locks down §2.
- **Exit:** CI green; both targets driven from pytest with no compiler yet.

### Phase 1: Frontend + Graph IR + oracle (1.5 weeks)
- Lexer, parser, AST, diagnostics with spans; Graph IR data structures, printer, parser for
  the textual IR (round trip), verifier, interpreter, `ref.py`.
- **Exit:** every example parses and type-checks; IR print → parse → print is stable; error
  messages are snapshot-tested; the interpreter equals the oracle (layers 1–2).

### Phase 2: Pass manager + graph passes (1 week)
- `PassManager` with `--print-after-all`, `--verify-each`, pass timing; shape inference,
  canonicalize, CSE, DCE, const-fold, fuse (matmul + bias/relu epilogue), range check,
  legalize per target.
- **Exit:** a unit test per pass (directed IR in, expected IR out); `--verify-each` is green on
  all examples.

### Phase 3: TPU backend (1.5 weeks)
- Tile IR, tiling lowering, tile passes, `.ttpu.json` emitter + JSON Schema, TPU cost model,
  `tpu_runner.py`.
- **Exit:** every TPU-legal example runs on the real RTL bit-exact; `big_mm.tiny` (12×12×12,
  27 passes) passes; the cost model equals measured cycles; golden + perf pins recorded
  (layers 1–3, 5, 6).

### Phase 4: GPU backend `-O0` (2 weeks)
- Kernel IR + interpreter, lowering (one kernel per op), memory planner, linear-scan regalloc,
  `.tgpu` emitter, `launch.json`, `gpu_runner.py` with counters.
- **Exit:** every example (including the multi-matmul `mlp_gpu.tiny`) runs on the simulator
  bit-exact; regalloc never touches R5–R7 (asserted); golden + perf pins (layers 1–3, 5, 6).

### Phase 5: GPU optimizations `-O1`/`-O2` (1.5 weeks)
- LICM, strength reduction, K-loop unrolling, shared-memory tiling (`SHLD/SHST/SYNCB`),
  fused epilogues.
- `docs/ARCHITECTURE.md` gets a measured table per optimization (instructions, cycles, global
  loads, for `mm_relu` 4×4 and 8×8).
- **Exit:** all opt levels bit-exact; `-O2` has strictly fewer global loads than `-O0` on every
  matmul example; perf pins updated on purpose.

### Phase 6: Differential fuzzing + hardening (1 week)
- Hypothesis generator over legal programs, both targets × all opt levels; regression corpus;
  nightly 5000-example job that uploads the shrunk failing `.tiny`.
- **Exit:** 5000 nightly examples green twice in a row; every bug found has a regression test.
  **At this point the CLI compiler can be published on its own.**

### Phase 7: Compiler Explorer web app (2.5 weeks)
- Pyodide worker + wheels, `useCompiler`, PassExplorer (L1) first, then L2 TPU schedule (reuse
  PEGrid), L3 GPU kernel, L4 head to head.
- **Exit:** compile + run works for both targets in the browser at desktop and 375 px, in light
  and dark; the Playwright smoke test passes; first-compile time under 5 s on a warm cache.

### Phase 8: Docs, landing page, launch (1 week)
- Landing page with a hero clip (one program, both chips side by side), docs pages
  (`/docs/language`, `/docs/ir`, `/docs/tpu-backend`, `/docs/gpu-backend`, `/docs/verification`),
  and a README in the series style with a verification table (fuzz examples, golden count,
  bit-exact on both targets).
- Deploy to Vercel (`tiny-compiler.vercel.app`); link from the tiny-gpu and tiny-tpu READMEs.
- **Exit:** public URL live; all CI green.

**Total: about 13 weeks part-time.** Phases 0–6 (about 9.5 weeks) give you a verified CLI
compiler for both chips.

---

## 11. Risks and mitigations

| Risk | Mitigation |
|---|---|
| tiny-gpu "cycles" undercount straight-line work (§2.1), which makes unrolling look miraculous | Always report instructions issued next to cycles; caveat in L4; directed test pins both numbers |
| int32 overflow raises inside tiny-gpu | `range_check` pass with interval analysis on every value; the fuzzer generates values near the limits |
| R7 = 0 after a grid resize | The runner constructs the simulator at its final size; contract test; upstream fix PR |
| Submodule bump changes the ISA or WASM API | Pinned commits + `test_targets_contract.py` runs first in CI |
| Pyodide bundle size / load time | Lazy load in a worker on the first Compile; only numpy as a package; tinygpu installed `--no-deps`; show progress |
| Python simulator is slow for big grids | UI caps sizes at 16×16; `history=False` for stats-only runs; tests use ≤ 12 |
| Two executors for `.ttpu` drift apart | There is only one (`ttpu-exec.mjs`); Python calls it through Node |
| Scope creep into a "real ML compiler" | v1 op set is fixed (§3.2); everything else is in §13 and starts only after Phase 8 |

---

## 12. CI matrix

| Workflow | Trigger | Jobs |
|---|---|---|
| `py.yml` | push/PR touching `tinyc/ runtime/ tests/ examples/ targets/` | ruff + black --check → contract → unit → e2e (both targets, all -O) → golden → perf pins → fuzz 200 |
| `fuzz-nightly.yml` | cron | fuzz 5000; upload shrunk `.tiny` + IR dumps on failure |
| `web.yml` | push touching `web/ runtime/` | build wheels → lint → typecheck → vitest → build → Playwright smoke |

**Full pre-PR check** (goes in `CLAUDE.md`):
```bash
source scripts/env.sh
ruff check . && black --check .
pytest -q                      # contract, unit, e2e, golden, perf, fuzz(200)
cd web && pnpm lint && pnpm typecheck && pnpm test && pnpm build
```

---

## 13. Stretch goals (after v1 ships)

1. **Requantization + MLPs** (`requant(x, shift) → i8`), which makes chained TPU matmuls legal
   (2-layer MLP on both chips). The GPU needs a shift: add `SHR`/`MAX`/`SUB` to tiny-gpu upstream
   and model them as **target features** (`--target tiny-gpu+shr`).
2. **tiny-cpu backend (RV32I)** once tiny-cpu's assembler lands: a scalar-loop baseline as a
   third column in L4. Then run the TPU schedule's `host.*` ops as RV32I code on tiny-cpu:
   the CPU orchestrates the TPU.
3. **Hardware/compiler co-design:** the cost model predicts the savings of a tiny-tpu
   `accumulate` mode (don't clear psum) and `keep_weights` (skip LOAD_WEIGHTS). Then implement
   both in tiny-tpu's RTL, expose them as target features, and check that the measurement matches
   the prediction.
4. **Autotuner:** search unroll factors, tile orders and shared-memory tiling with the cost
   model, then confirm the winners on the real targets.
5. **conv2d via im2col** lowering onto the same matmul paths.
6. **`docs/MLIR_MAPPING.md`:** map Graph IR → `linalg`, Tile IR → `scf`/`affine` + a custom
   dialect, Kernel IR → `gpu`. It explains how the toy maps onto a production compiler.
7. **tiny-attention target**, if that project ships a hardware interface.

---

## 14. First week, concretely

1. Create `deaneeth/tiny-compiler`; add the submodules `targets/tiny-gpu`, `targets/tiny-tpu`;
   copy `.editorconfig` and the web config from tiny-tpu; write CLAUDE/AGENTS/CONTEXT.
2. `scripts/env.sh` + venv; `pip install -e targets/tiny-gpu --no-deps`.
3. `runtime/gpu_runner.py` running tiny-gpu's `vector_add.tgpu` from pytest.
4. `runtime/ttpu-exec.mjs` + `ttpu-cli.mjs` running one 4×4 pass on `tiny_tpu.wasm`, called
   from pytest.
5. `test_targets_contract.py` for every fact in §2, then the lexer for §3.2.
