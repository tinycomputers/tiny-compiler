# Verification

Every claim the compiler makes is checked against something independent. Layers, from the source out:

| Layer | What | Where |
|---|---|---|
| Oracle | numpy int64 evaluation of the *source* program; any value outside i32 is an error (tiny-gpu raises on overflow) | `tinyc/ir/ref.py` |
| Interpreters | independent scalar interpreters for Graph IR, the TPU command list, and Kernel IR (with exact tiny-gpu counts, including barriers) | `ir/interp.py`, `tpu/interp.py`, `gpu/interp.py` |
| Per-pass validation | `--verify-each` re-runs the program after *every* pass and compares with the oracle | `passes/manager.py`, `tpu/__init__.py`, `gpu/__init__.py` |
| Real targets | emitted code runs on the unmodified tiny-gpu simulator and the Verilated tiny-tpu RTL | `runtime/` |
| Contract tests | every fact about the targets the design depends on is measured and pinned | `tests/test_targets_contract.py` |
| Properties | Hypothesis: semantics preserved, pipeline idempotent, range-check sound, estimates bound measurements | `tests/**/test_*properties.py`, `tests/passes/test_pipeline.py` |
| Differential fuzzing | random programs through *every* backend and level on the real chips | `tests/test_fuzz.py`, `tests/fuzz_support.py` |
| Regression corpus | corner cases and past bugs, replayed everywhere | `tests/regressions/*.tiny` |
| Goldens & pins | emitted IR/code/diagnostics; cost numbers per example and level | `tests/golden/`, `tests/perf_baseline.json` |
| Docs that quote numbers | the measured optimization table must equal a fresh measurement | `tests/gpu/test_opt_table.py` |

## Differential fuzzing
`check_program` compiles one program for `gpu -O0/-O1/-O2` and `tpu -O0/-O1`, runs each on the real chip,
and requires bit-exact agreement with the oracle, R5-R7 untouched, estimate bounds containing the
measurement (loads/stores exact), and the TPU cost model equal to the measured cycles. A program a
backend legitimately rejects (a chained matmul on the TPU, an overflow that cannot be ruled out) is
skipped for that backend only.

```bash
pytest tests/test_fuzz.py                                   # 100 examples, fixed seed (part of CI)
FUZZ_EXAMPLES=5000 FUZZ_RANDOM=1 pytest tests/test_fuzz.py  # the nightly run: fresh programs
```
On failure Hypothesis shrinks the program and it is left in `tests/regressions/_last_failure.tiny`
(the nightly workflow uploads it). Understand the bug, fix it, rename the file to describe the bug and
commit it: `tests/test_regressions.py` then replays it on every backend forever.

## Bugs the verification found while building
Each is covered by a regression test.
- The simplification passes were not idempotent (CSE broke canonical operand order); fixed by running
  them to a fixpoint. `tests/regressions/duplicate_subexpressions.tiny` and `test_manager.py`.
- A pass-generated instruction had no source position (strength reduction's pointer init).
- Fusion was silently untested because the first random generator never produced a fusable pattern.
- Early tests pinned cycle counts the simulator does not produce; measuring first fixed the model.
