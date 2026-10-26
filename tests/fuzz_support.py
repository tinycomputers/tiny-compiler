"""One differential check, used by the fuzzer and the regression corpus.

A program is compiled for every backend and optimization level that can run it, executed on the
real chip/simulator, and every result must equal the numpy oracle computed from the *source*
program (before any pass). Two backends therefore agree with each other by transitivity.
"""

from __future__ import annotations

import numpy as np

from runtime.gpu_runner import run_launch
from runtime.tpu_runner import run_ttpu
from tinyc.driver import build_pipeline, compile_gpu, compile_tpu
from tinyc.frontend import compile_to_graph
from tinyc.ir import ref
from tinyc.ir.errors import PassError, RangeError
from tinyc.passes import PassContext, PassManager

GPU_LEVELS = (0, 1, 2)
TPU_LEVELS = (0, 1)


def check_program(source: str, seed: int = 1) -> dict[str, str]:
    """Raises AssertionError (naming target and level) on any disagreement.

    Returns {backend-and-level: "ok" | "rejected: why"} so callers can see what actually ran.
    """
    func = compile_to_graph(source, "fuzz.tiny")
    inputs = ref.random_inputs(func, np.random.default_rng(seed))
    try:
        want = ref.evaluate(func, inputs)
    except RangeError:
        return {"source": "overflows i32 on these inputs"}

    ran: dict[str, str] = {}
    for opt in GPU_LEVELS:
        key = f"gpu-O{opt}"
        try:
            comp = compile_gpu(source, "fuzz.tiny", opt_level=opt)
        except PassError as e:
            ran[key] = f"rejected: {e.message}"
            continue
        res = run_launch(comp.launch, inputs)
        assert res.reserved_intact, f"{key}: a kernel clobbered R5-R7"
        for name, w in zip(comp.launch["outputs"], want):
            assert np.array_equal(
                res.outputs[name], w
            ), f"{key}: output %{name} differs from the oracle"
        est = comp.program.estimate
        assert est["instructions"]["min"] <= res.instructions <= est["instructions"]["max"], key
        assert est["cycles"]["min"] <= res.cycles <= est["cycles"]["max"], key
        assert (res.op_counts["LD"], res.op_counts["ST"]) == (est["loads"], est["stores"]), key
        ran[key] = "ok"
    for opt in TPU_LEVELS:
        key = f"tpu-O{opt}"
        try:
            comp = compile_tpu(source, "fuzz.tiny", opt_level=opt)
        except PassError as e:
            ran[key] = f"rejected: {e.message}"
            continue
        res = run_ttpu(comp.program, inputs)
        for name, w in zip(comp.schedule.outputs, want):
            assert np.array_equal(
                res.outputs[name], w
            ), f"{key}: output %{name} differs from the oracle"
        est = comp.program["estimate"]
        assert res.total_cycles == est["rtl_cycles"], f"{key}: cost model != measurement"
        assert res.host_ops == est["host_ops"], key
        ran[key] = "ok"
    return ran


def pipeline_is_idempotent(source: str) -> bool:
    """Running the graph pipeline on its own output changes nothing."""
    func = compile_to_graph(source, "fuzz.tiny")
    try:
        PassManager(build_pipeline(None, 1)).run(func, PassContext())
    except PassError:
        return True  # rejected by range-check: nothing to compare
    again = PassManager(build_pipeline(None, 1)).run(func, PassContext())
    return not any(r.changed for r in again.records)
