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
- Next: Phase 2 (pass manager + graph passes).

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
| TPU executor | Exactly one: `runtime/ttpu-exec.mjs` (Node in tests, browser in UI) |
| Overflow | Range analysis must prove i32 fits; otherwise reject (tiny-gpu raises on overflow) |
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
