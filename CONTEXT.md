# CONTEXT.md - TinyCompiler

Read this before any task. Full plan: `docs/PLAN.md`.

## What this is
A small multi-level compiler for a tiny tensor language (`.tiny`). One program, e.g.
`C = relu(A @ B)`, lowers to tiny-gpu SIMT assembly (`.tgpu` + `launch.json`) or to a tiny-tpu
4×4 tile schedule (`.ttpu.json`), then runs on the unmodified tiny-gpu simulator or the real
tiny-tpu RTL (WASM). Fourth project after tiny-gpu, tiny-tpu and tiny-cpu. Nothing on screen is
faked; host-side work is always labelled as host work.

## Status
- Phase 0 (scaffold + drive both targets from pytest): DONE. 17 tests green (runners + target contract), ruff/black clean.
- Phase 1 (frontend + Graph IR + oracle): DONE. 90 tests green: lexer/parser/diagnostics (20 golden error messages), Graph IR
  printer/parser round trip, verifier, numpy oracle + independent scalar interpreter, 6 examples.
- Phase 2 (pass manager + graph passes): DONE. Passes: shape-infer, FIXPOINT{canonicalize, const-fold, cse, dce},
  fuse (matmul{epilogue=bias,relu}), range-check, legalize-tpu/gpu; `--verify-each`, `--print-after-all`, `--time-passes`.
- Phase 3 (TPU backend): DONE. 287 tests green. Graph IR -> Tile IR (`tinyc/tpu/`) -> `.ttpu.json` (schema `docs/ttpu.schema.json`,
  spec `docs/TTPU_FORMAT.md`); `runtime/ttpu-exec.mjs` runs whole programs on the real RTL; cost model == measurement (pinned
  in `tests/perf_baseline.json`). `tinyc FILE --target tpu --run` works.
- Phase 4 (GPU backend, -O0): DONE. 443 tests green. Graph IR -> memory plan -> Kernel IR -> linear-scan registers -> `.tgpu` + launch JSON
  (`tinyc/gpu/`, schema `docs/launch.schema.json`, spec `docs/TGPU_FORMAT.md`). Every example, including the chained-matmul
  `mlp_gpu`, runs bit-exact on the real simulator; the same program gives identical results on both chips. `tinyc FILE --target gpu --run`.
- Next: Phase 5 (GPU optimizations -O1/-O2: LICM, strength reduction, unrolling, shared-memory tiling). `-O` currently
  controls only the graph passes; decide how it maps to kernel optimizations before adding them.

## Fixed decisions (do not re-litigate; change only via docs/ARCHITECTURE.md)
| Decision | Choice |
|---|---|
| Language | `.tiny`, static shapes, rank ≤ 2, types `i8`/`i32`, ops `@`, `+` (row broadcast), `relu` |
| Implementation | Python 3.12 + numpy; runs in the browser via Pyodide (no TS re-implementation) |
| IR levels | Graph IR (SSA, typed, loc-tagged) → Tile IR (TPU) / Kernel IR (GPU); printer + verifier + interpreter for each |
| Targets | Submodules at pinned commits; v1 does not modify either chip |
| GPU ABI | R5 = block_id, R6 = thread_in_block, R7 = tid are reserved; addresses baked in as immediates; no kernel args |
| GPU mapping | One thread per output element: block = row i, thread = column j |
| GPU regalloc | Linear scan, budget 16 by default, no spilling (compile error) |
| TPU mapping | 4×4 int8 passes, zero padding, K-accumulation + relu/bias as `host.*` ops |
| TPU executor | Exactly one: `runtime/ttpu-exec.mjs` (Node in tests, browser in UI). `tinyc/tpu/interp.py` is a second, independent *reference* reader of the same JSON spec, never a second executor |
| TPU host work | `acc`/`epilogue`/`host_add`/`host_relu`/`alloc` are host commands, counted separately from chip passes everywhere |
| GPU mapping | One thread per output element: block = row (R5), thread = column (R6), tid (R7) = flat index. One kernel per Graph op (fused = one). Kernel code is read-only on R5-R7 (asserted + checked at run time) |
| GPU cycles | A thread needs taken-branches+1 cycles, a launch the max. Report instructions with cycles. `estimate` gives min/max (relu is data dependent), loads/stores exact; `gpu.interp` counts must equal the simulator's |
| GPU asm rules | Labels are `L<n>` (a label starting with R is parsed as a register); a kernel never ends with an `if` (stray extra cycle); one fresh simulator per launch at its final size, memory carried over |
| TPU cost | `passes * 15` RTL cycles; utilization = useful MACs (`M*K*N`) / (64 * passes) |
| Overflow | `range-check` proves every value fits i32 by intervals; otherwise reject (tiny-gpu raises on overflow). i32 *inputs* are assumed within ±2^20 (`PassContext.i32_input_range`); runners must respect it |
| Fusion | `matmul{epilogue=bias,relu}`; the fused op takes the outermost absorbed op's loc. Backends must lower the epilogue |
| Verification | Oracle (numpy) + per-pass translation validation + real targets + Hypothesis differential + golden + perf pins |

## Target gotchas (see PLAN §2)
- tiny-gpu runs straight-line instructions in one "cycle": report instructions issued as well.
- tiny-gpu `set_grid` resize leaves R7 = 0 for threads it adds: build the simulator at its final size.
- tiny-tpu passes always reload weights and start from psum = 0; `getCycleCount()` reports 15 per pass (14 + final latch).
- tiny-tpu silently truncates operands to int8: the compiler must reject non-i8 itself.
- Never run `black .`/formatters over `targets/` (submodules); pyproject excludes it.
- Python deps: `pip install -e targets/tiny-gpu --no-deps` (tiny-gpu's matplotlib/imageio are not needed).

## Conventions
- Python: 4-space indent, Black + Ruff, tests under `tests/`, fixed seed 42.
- Web: Astro + React islands + Tailwind, TypeScript strict; WASM/Pyodide only in
  `client:only="react"` islands and workers.
- `docs/TTPU_FORMAT.md` and its JSON Schema must stay in sync with `tinyc/tpu/emit.py`.
