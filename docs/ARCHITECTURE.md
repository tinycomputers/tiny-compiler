# Architecture

How the compiler is put together, and the decisions that are not obvious from the code. Change a
fixed decision here first (see `CONTEXT.md`). Per-pass details are in [PASSES.md](PASSES.md); the
formats are in [LANGUAGE.md](LANGUAGE.md), [IR.md](IR.md), [TTPU_FORMAT.md](TTPU_FORMAT.md) and
[TGPU_FORMAT.md](TGPU_FORMAT.md).

## Pipeline

```
 .tiny -> AST -> Graph IR -> graph passes -+-> Tile IR -> tile passes -> .ttpu.json   (tiny-tpu RTL)
                                           |
                                           +-> memory plan -> Kernel IR -> kernel passes
                                                            -> registers -> .tgpu + launch  (tiny-gpu)
```

Every level has a printer, a verifier and an interpreter; every pass can be validated against the
numpy oracle (`--verify-each`); and both backends run on the real chip/simulator, never on a
re-implementation.

## Optimization levels (`-O`)

| Level | Graph IR | Tile IR (tpu) | Kernel IR (gpu) |
|---|---|---|---|
| `-O0` | shape-infer, range-check | none | naive kernels |
| `-O1` (default) | + simplify fixpoint, fuse | + skip-zero-passes | + licm, strength-reduce, kernel-dce |
| `-O2` | same as `-O1` | same as `-O1` | + shared-tile, unroll, fold |

Individual passes can be switched off by name (`--disable licm`), which is how the table below
isolates each optimization.

## Kernel optimizations, measured

Real simulator runs (`tinyc/gpu/report.py`; regenerate with `python scripts/opt_table.py --write`,
and a test fails if this table drifts). "Cycles" are tiny-gpu cycles: a thread's cycle ends at a taken
branch, so straight-line code is one cycle however long it is; read them together with instructions.
`relu` makes the exact counts data dependent, so these use a fixed seed.

<!-- opt-table:start -->
**`mm_relu` 4x4** (`C = relu(A @ B)`, int8, seed 42)

| configuration | instructions | cycles | global loads | shared loads | registers |
|---|---:|---:|---:|---:|---:|
| -O0 (naive) | 1000 | 6 | 144 | 0 | 4 |
| + licm | 840 (-16%) | 5 | 128 (-11%) | 0 | 5 |
| + strength-reduce (= -O1) | 728 (-27%) | 5 | 128 (-11%) | 0 | 6 |
| -O2 with shared-memory tiling only | 728 (-27%) | 6 | 80 (-44%) | 64 | 5 |
| -O2 with unrolling + folding only | 504 (-50%) | 2 | 128 (-11%) | 0 | 7 |
| -O2 (all passes) | 504 (-50%) | 3 | 80 (-44%) | 64 | 5 |

**`mm_relu` 8x8** (`C = relu(A @ B)`, int8, seed 42)

| configuration | instructions | cycles | global loads | shared loads | registers |
|---|---:|---:|---:|---:|---:|
| -O0 (naive) | 7320 | 10 | 1088 | 0 | 4 |
| + licm | 6168 (-16%) | 9 | 1024 (-6%) | 0 | 5 |
| + strength-reduce (= -O1) | 5208 (-29%) | 9 | 1024 (-6%) | 0 | 6 |
| -O2 with shared-memory tiling only | 4952 (-32%) | 10 | 576 (-47%) | 512 | 5 |
| -O2 with unrolling + folding only | 3544 (-52%) | 2 | 1024 (-6%) | 0 | 7 |
| -O2 (all passes) | 3288 (-55%) | 3 | 576 (-47%) | 512 | 5 |

<!-- opt-table:end -->

What each pass does, and what it costs:

- **licm** hoists address arithmetic that does not depend on the loop counter (`row*K + base`).
  It never hoists loads, so global loads are unchanged. It *raises register pressure*: hoisted values
  stay live across the loop, which is why `-O0` is the code that fits the smallest register file.
- **strength-reduce** turns `k*N + col + base` (a multiply and two adds per iteration) into a pointer
  initialised once and bumped by `N`. Chains of one statement are left alone: an add costs the same as
  the increment that would replace it.
- **shared-tile** exploits that every thread of a block reads the same row of A. The block's threads
  copy it into shared memory cooperatively, meet at a barrier (`SYNCB`), and read it with `SHLD`.
  Global loads per matmul drop from `2*M*N*K` to `M*K + M*N*K` (about -44% to -47% above). It is roughly
  instruction-neutral: the copy and the barrier cost about what the cheaper `shld` (no address
  arithmetic) saves, and in loop form it adds a cycle. Skipped for one-thread blocks.
- **unroll** removes the loop for trip counts up to 16, so the kernel has no taken back edges and its
  cycle count collapses. **fold** then folds the constant addresses and copies that unrolling exposes.

### Why -O2 is not simply better

- Unrolling alone is the biggest instruction and cycle win but the most register-hungry (7 registers
  above: every iteration's temporaries are live at once before the allocator packs them); tiling
  *reduces* pressure back to 5 because the A address chain disappears. The default register file (13
  allocatable) has room, but `--gpu-regs 8` only fits `-O0`.
- It needs a barrier, and tiny-gpu re-executes `SYNCB` on every step a thread spends waiting, so the
  instruction count includes those repeats (modelled exactly in `gpu/interp.py`).
- Tiling only pays when a block has at least two threads; with a reduction as short as `K = 2` the
  copy and barrier are not amortised, so loads and cycles improve more than instructions.
- A kernel tiles `K` shared words per operand per block; the simulator allocates exactly that.

## Decisions worth knowing

| Decision | Why |
|---|---|
| Temporaries are assigned once; only accumulators, pointers and counters are reassigned | It makes every kernel pass sound without dataflow analysis: a single-assignment value can be moved or rewritten freely. |
| Kernel passes are validated against the oracle after each pass | The structured kernels are interpreted (independently of the allocator and assembler); the final assembly is run on the simulator. |
| The estimate is an abstract interpretation | Values that depend only on block/thread ids are exact, so loads, stores, shared accesses and the tiled copy's guards are exact; only a branch on loaded data (relu) gives `min`/`max`. |
| One fresh simulator per kernel launch, memory carried over | tiny-gpu's `set_grid` resize leaves R7 = 0 for threads it adds. |
| `-O` levels are cumulative, kernel passes are switchable by name | Reproducibility, and so the table above can attribute each gain. |

## Known limits

- Kernel optimization is specific to what `lower.py` emits (matmul-shaped loops); the passes are
  written generally but only exercised on those kernels.
- No spilling: a program that needs more registers than the budget is a compile error.
- The dead final pointer bump of an unrolled loop is not removed (the DCE is not flow-sensitive).
