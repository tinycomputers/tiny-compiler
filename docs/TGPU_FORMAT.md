# tiny-gpu output: `.tgpu` kernels and the launch document

`tinyc FILE --target gpu --emit tgpu` prints the assembly; `--emit launch` prints the launch
document (machine-checked by `docs/launch.schema.json`); `--emit kernel` prints the Kernel IR.

## Mapping
One thread per output element. A rank-2 result `[M, N]` launches **M blocks of N threads**:
`R5` (block id) is the row, `R6` (thread in block) is the column, and `R7` (global thread id) is
the flat row-major index. Rank-1 results use one block. Each Graph IR op is one kernel launch
(a fused `matmul{epilogue=...}` is one kernel); launches share global memory.

`R5`, `R6`, `R7` are set by the simulator and **only ever read**. Kernel code uses
`R0-R4, R8..R(n-1)`; an assertion in the allocator and a runtime check in the runner
(`reserved_intact`) enforce this. Nothing is passed as kernel arguments: all addresses are
immediates from the memory plan.

## What tiny-gpu has and does not have
`SET ADD MUL LD ST CMP BRLT BRGT` (and `JMP BEQ BNE BRZ SYNC SYNCB SHLD SHST CSWAP`). There is no
SUB, DIV, shift or MAX, so:
- loops count up: `SET k,0 / L: ... / ADD k,k,1 / CMP k,K / BRLT L`;
- `relu(x)` is `CMP x,0 / BRGT skip / SET x,0 / skip:`;
- there is no base+offset addressing: every access computes its address in a register.

Assembler traps found by measurement (and enforced by the emitter):
- **a label must not start with `R`/`r`**: tiny-gpu's assembler parses it as a register and
  crashes (`relu_end:` -> `ValueError`). Labels here are `L0`, `L1`, ...;
- **a kernel must not end with an `if`**: branching to the end of the program costs a stray extra
  cycle; the Kernel IR verifier rejects it.

## Shared memory and barriers (`-O2`)
`SHST saddr, Rs` / `SHLD Rd, saddr` access the block's own shared memory (`shared_size` words per
block, declared per kernel in the launch document); `SYNCB` is a block-wide barrier. Measured and
modelled behaviour:
- a thread that reaches `SYNCB` stops for that step, and a **waiting thread executes `SYNCB` again on
  every following step** until all threads of its block are waiting; then the block resumes on the next
  step. With arrival step `s_t = taken branches before the barrier + 1` and release step `R = max s_t`,
  a thread executes `SYNCB` `R - s_t + 1` times and finishes at step `R + taken branches after + 1`.
  `gpu/interp.py` implements this and a test requires it to equal the simulator exactly, including for
  hand-written kernels whose threads arrive at different times;
- out-of-range `SHLD` silently reads 0 and out-of-range `SHST` is silently dropped in tiny-gpu; the
  interpreter treats both as faults, so a compiler bug cannot hide;
- the Kernel IR only allows a barrier at the top level of a kernel, so every thread reaches it.

## Cycles vs instructions
tiny-gpu runs consecutive non-branch instructions of a thread in one step; a thread's step ends at
a *taken* branch. A thread needs `taken branches + 1` cycles, a launch the maximum over threads. So
a straight-line kernel is 1 cycle however long it is, and a K-iteration loop is K cycles.
**Always read cycles together with instructions.** The relu `if` is data dependent: a thread that
clamps runs one more instruction (`SET`); a thread that does not takes the skip branch, which adds a
cycle. The static `estimate` is an abstract interpretation: anything that depends only on block/thread ids is
exact (loop trip counts, a tiled kernel's copy guards), so loads, stores and shared accesses are exact;
only a branch on loaded data gives `min`/`max` for instructions and cycles. `gpu.interp` gives exact counts for given inputs, and tests require them to equal
the simulator's.

## Launch document
```json
{ "format": "tgpu-launch", "version": 1, "name": "main", "num_registers": 16, "mem_size": 140,
  "buffers": { "X": {"addr": 0, "dtype": "i8", "shape": [4,8], "kind": "input"},
               "2": {"addr": 104, "dtype": "i32", "shape": [4,6], "kind": "temp"} },
  "inputs": ["X"], "outputs": ["3"],
  "kernels": [ {"name": "k0_matmul", "value": "2", "op": "matmul", "grid": [4,6],
                "shared_size": 0, "asm": "SET R0, 0 ; ..."} ],
  "estimate": { ... } }
```
Memory is word addressed (`i8` takes one int32 word). Constants (`kind: const`, with `data`) and
inputs are the initial memory image; everything else starts at 0. Intermediates reuse the words of
buffers whose last reader has finished (`gpu/memplan.py`); an op's result never overlaps what that
same op reads.

## Executor
`runtime/gpu_runner.py::run_launch` runs a launch document on the real simulator: a fresh
`TinyGPU` per kernel built at its final thread count (the simulator's `set_grid` resize leaves
R7 = 0 for added threads), global memory carried over. It counts instructions per opcode without
changing semantics.

## Source mapping
Each instruction ends with `; line:col`, the source position of the Graph IR op it came from.
