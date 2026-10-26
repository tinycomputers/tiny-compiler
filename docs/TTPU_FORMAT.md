# `.ttpu.json` (version 1)

The program a tiny-tpu run executes. Machine-checked by `docs/ttpu.schema.json`
(`tests/tpu/test_ttpu_format.py` validates every emitted program against it).
Consumers: `runtime/ttpu-exec.mjs` (the executor; Node and browser) and `tinyc/tpu/interp.py`
(an independent reference interpreter written from this spec).

```json
{ "format": "ttpu", "version": 1, "tile": 4, "name": "main",
  "buffers": { "A": {"dtype":"i8","shape":[4,8],"kind":"input"},
               "1": {"dtype":"i32","shape":[4,4],"kind":"temp"} },
  "inputs": ["A", "B"], "outputs": ["1"],
  "commands": [ ... ], "estimate": { ... } }
```

## Buffers
`dtype` is `i8` or `i32`; `shape` has rank 1 or 2 (row-major). `kind`:
`input` (supplied at run time), `const` (carries `data` as nested lists) or `temp` (zero-filled
at start). Buffer names are the Graph IR value names without `%`.

## Commands, in execution order
Only `pass` runs on the chip. **Everything else is host work**, counted separately and never
attributed to the chip.

A *window* `{"buf", "r", "c"}` is the 4x4 region of a rank-2 buffer whose top-left element is
`(r, c)`. Reading outside the buffer yields 0 (zero padding); writing outside it is dropped.

| `op` | Fields | Meaning |
|---|---|---|
| `alloc` | `buf` | **host.** Zero the buffer. |
| `pass` | `id`, `a`, `b` (windows) | **chip.** `tmp[id] = a_window @ b_window` (int8 x int8 -> int32, 4x4). Each pass starts from psum = 0 and reloads its weights. |
| `acc` | `pass`, `dst` (window), optional `bias` (buffer name), optional `relu: true` | **host.** `dst += tmp[pass]` (in-bounds part), then the epilogue: `dst[r][c] += bias[c]`, then `max(dst, 0)`. |
| `epilogue` | `dst` (window), optional `bias`, `relu` | **host.** The epilogue alone (an output tile that no pass feeds). |
| `host_add` | `dst`, `a`, `b` | **host.** Whole-buffer add; same shape, or a rank-1 operand broadcast over the rows of a rank-2 one. |
| `host_relu` | `dst`, `src` | **host.** `dst = max(src, 0)`. |

Every pass is consumed by exactly one `acc`, after it ran. Every value must fit in i32 or the
executor fails (`RangeError`); `range-check` proves this at compile time for any legal input.

## Tiling order
For `C = A @ B`, `A: [M,K]`, `B: [K,N]`: output tiles row-major, K innermost, matching tiny-tpu's own
`tilingSchedule()`. `ceil(M/4) * ceil(N/4) * ceil(K/4)` passes before optimization. The last `acc`
of an output tile carries that tile's epilogue.

## `estimate`
Static cost model; the executor must measure exactly this (`tests/tpu/test_tpu_rtl.py`).

| Key | Meaning |
|---|---|
| `passes`, `passes_skipped` | chip passes left, and passes removed by `skip-zero-passes` |
| `cycles_per_pass` | 15 (14 from start to done, +1 for the last element to latch; `getCycleCount()`) |
| `rtl_cycles` | `passes * 15` |
| `host_ops` | counts per host command kind |
| `useful_macs` / `capacity_macs` / `utilization` | non-padding multiply-adds (`M*K*N` for the passes kept) over `64 * passes` (a pass is 4x4x4 MACs) |

## Executor protocol (`runtime/ttpu-cli.mjs`)
Request `{ "program": <this document>, "inputs": { name: nested lists } }` ->
`{ "outputs": { name: {shape, data} }, "passes": [{id, cycles}], "totalCycles", "hostOps" }`.
