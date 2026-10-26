# Graph passes and the pass manager

`tinyc/driver.py` builds the pipeline; `tinyc/passes/manager.py` runs it.

```
-O1:  shape-infer, FIXPOINT{canonicalize, const-fold, cse, dce}, fuse, range-check, [legalize-<target>]
-O0:  shape-infer, range-check, [legalize-<target>]
```

The four simplifications run as a **fixpoint group** (`passes.manager.Fixpoint`): they enable each
other, so the whole group repeats until a full round changes nothing. A single sweep is not enough:
CSE can merge an op into an earlier one and leave a commutative `add` with its operands in
non-canonical order (Hypothesis found this; `tests/passes/test_manager.py` keeps the shrunk
program). Every sub-pass is still verified, recorded and dumped on every round, so
`--print-after-all` shows each round.

| Pass | What it does |
|---|---|
| `shape-infer` | Recomputes every result type from its operands (repairs parsed or hand-edited IR). |
| `canonicalize` | Broadcast `add` puts the matrix first; same-shape `add` orders operands by definition order; `relu(relu(x))` becomes `relu(x)`. |
| `const-fold` | Evaluates ops whose operands are all constants, using the oracle's own `eval_op`. A constant expression that overflows i32 is a compile error. |
| `cse` | Merges identical ops and identical constants (same data **and** dtype). |
| `dce` | Removes ops that never reach an output. Inputs are always kept. |
| `fuse` | `matmul -> add(bias[N]) -> relu` becomes one `matmul{epilogue=bias,relu}`; any prefix works (`bias`, `relu`, `bias,relu`). A stage is absorbed only if the previous result has exactly one use and is not an output. The fused op sits where the last absorbed op was and takes **that op's source location** (the outermost expression). |
| `range-check` | Interval analysis: proves every value fits in i32 or rejects the program (below). |
| `legalize-tpu` | Every matmul operand must be an i8 input or constant. |
| `legalize-gpu` | Every op result needs at most `gpu_max_threads` (1024) threads; total buffers must fit `gpu_mem_words` (a conservative bound). |

## Range analysis
- i8 inputs are exactly `[-128, 127]`. **i32 inputs are assumed to lie in `ctx.i32_input_range`
  (default ±2^20)**: an i32 input can otherwise hold any 32-bit value, and then nothing involving it
  could be proven. Whatever runs a compiled program must respect that assumption.
- Constants are exact. A matmul of K products each in `[pmin, pmax]` lies in `[K*pmin, K*pmax]`;
  prefix sums stay inside the hull of that interval and 0, so the proof holds for any accumulation
  order (GPU loop, TPU tiling).
- The result is stored in `ctx.analysis["ranges"]` as `{value name: (lo, hi)}`.

## Verification
- After **every** pass the IR is structurally verified (always on).
- `--verify-each` also *runs* the IR after every pass on seeded random inputs (the scalar
  interpreter) and compares with the numpy oracle's answers for the program before any pass. A
  failure raises `PassVerifyError` naming the pass, with the IR before and after.
- Tests: a directed IR-in/IR-out test per pass; the manager against a deliberately wrong pass;
  every example through every target and `-O` level; Hypothesis over random programs checks that
  the pipeline preserves semantics, is idempotent, and that range-check is *sound* (accepted
  programs never overflow on interval-corner inputs, and every value stays inside its interval).
- The generator emits dense-layer idioms on purpose: a plain random generator never produced a
  fusable pattern, so fusion was silently untested.

## CLI
```
tinyc examples/mm_bias_relu.tiny --target gpu --print-after-all --verify-each --time-passes
tinyc examples/mm_relu.tiny -O0                 # no optimizations
tinyc examples/mm_relu.tiny --disable fuse      # switch a pass off
```
Dumps and timings go to stderr; the final IR goes to stdout. Exit codes: 0 ok, 1 compile or
legalization error, 2 usage error, 3 internal compiler error (a pass broke the program).

## Tile passes (tiny-tpu backend)
After lowering to Tile IR (`tinyc/tpu/lower.py`) the `-O1` pipeline runs `skip-zero-passes`:
a pass whose A or B window is a constant all-zero tile is removed together with its accumulation
(its product is zero). If that accumulation carried the output tile's epilogue, the epilogue moves
to the previous kept accumulation, becomes a standalone `epilogue` (bias present), or disappears
(relu only: relu of the zero-initialized tile is a no-op). Pass ids are renumbered. `--verify-each`
validates the schedule after lowering and after each tile pass, by interpreting the emitted JSON.

## GPU backend stages
After the graph pipeline, `compile_gpu_program` runs: memory plan (`gpu/memplan.py`) -> lowering to
Kernel IR (`gpu/lower.py`, one kernel per op) -> kernel passes (`gpu/kernel_passes.py`) -> verify ->
flatten -> linear-scan register allocation (`gpu/regalloc.py`) -> `.tgpu` emission (`gpu/emit.py`).

| `-O` | Kernel passes |
|---|---|
| 0 | none (the naive kernels) |
| 1 | `licm`, `strength-reduce`, `kernel-dce` |
| 2 | `licm`, `strength-reduce`, `shared-tile`, `unroll`, `fold`, `kernel-dce` |

- `licm`: hoist single-assignment arithmetic that does not change inside a loop (never loads).
- `strength-reduce`: replace a `mul`/`add` chain on the loop counter by a pointer bumped each iteration
  (chains of two or more statements only; not when the value is read after the loop).
- `kernel-dce`: remove values that never reach a store, `if` or loop control (liveness-based, so a dead
  value that only updates itself goes too).
- `shared-tile`: a loop load whose address is `u + k`, with `u` the same for the whole block, is copied
  cooperatively into shared memory (`SHST`), a barrier (`SYNCB`) follows, and the loop reads `SHLD`.
- `unroll`: full unroll of innermost loops of at most 16 iterations (the counter becomes a constant).
- `fold`: constant propagation, constant folding, add-chain reassociation and copy propagation.

With `--verify-each` the structured kernels are interpreted against the numpy oracle after lowering
and after *every* kernel pass, and the emitted assembly is run on the real simulator against the
oracle. `--print-after-all` shows `lower-to-kernel`, each kernel pass, and `regalloc-emit`. Each pass
can be disabled by name. The measured effect of each is in [ARCHITECTURE.md](ARCHITECTURE.md).
