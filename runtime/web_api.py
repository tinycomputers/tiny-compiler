"""The browser-facing API: one call compiles for both chips, runs the tiny-gpu simulator, and
computes the numpy oracle. The tiny-tpu program is returned for the browser to run on its WASM RTL.

Runs under Pyodide (web/public/compiler-worker.js) and under CPython (tests). Returns a JSON string.
"""

from __future__ import annotations

import json

import numpy as np

from runtime.gpu_runner import run_launch
from tinyc.driver import compile_gpu, compile_tpu
from tinyc.frontend import CompileError, compile_to_graph
from tinyc.ir import ref
from tinyc.ir.errors import PassError, RangeError
from tinyc.passes import PassContext
from tinyc.tpu.tile_ir import print_schedule

FILENAME = "program.tiny"


def _render(e: Exception) -> str:
    if isinstance(e, CompileError):
        return e.render()
    if isinstance(e, PassError):
        return f"{e.loc}: error: {e.message}" if e.loc else f"error: {e.message}"
    return f"internal error: {type(e).__name__}: {e}"


def _gpu(source: str, opt: int, inputs, want) -> dict:
    dumps: list[dict] = []
    try:
        comp = compile_gpu(
            source,
            FILENAME,
            opt_level=opt,
            ctx=PassContext(),
            dump=lambda name, text: dumps.append({"name": name, "ir": text}),
        )
    except Exception as e:  # noqa: BLE001 - any failure is shown to the user, never a crash
        return {"ok": False, "error": _render(e), "passes": dumps}
    launch = comp.launch
    est = comp.program.estimate
    out = {
        "ok": True,
        "passes": dumps,
        "asm": comp.program.asm_text(),
        "kernels": [k["name"] + " " + str(k["grid"]) for k in launch["kernels"]],
        "launch": launch,
        "estimate": est,
        "run": None,
    }
    if inputs is not None and want is not None:
        res = run_launch(launch, inputs)
        exact = all(
            np.array_equal(res.outputs[name], w) for name, w in zip(launch["outputs"], want)
        )
        out["run"] = {
            "outputs": {n: res.outputs[n].tolist() for n in launch["outputs"]},
            "exact": bool(exact),
            "reservedIntact": bool(res.reserved_intact),
            "cycles": res.cycles,
            "instructions": res.instructions,
            "loads": res.op_counts["LD"],
            "stores": res.op_counts["ST"],
            "shared": res.op_counts["SHLD"] + res.op_counts["SHST"],
            "barriers": res.op_counts["SYNCB"],
        }
    return out


def _tpu(source: str, opt: int) -> dict:
    dumps: list[dict] = []
    try:
        comp = compile_tpu(
            source,
            FILENAME,
            opt_level=opt,
            dump=lambda name, text: dumps.append({"name": name, "ir": text}),
        )
    except Exception as e:  # noqa: BLE001 - any failure is shown to the user, never a crash
        return {"ok": False, "error": _render(e), "passes": dumps}
    return {
        "ok": True,
        "passes": dumps,
        "tile": print_schedule(comp.schedule),
        "program": comp.program,
        "estimate": comp.program["estimate"],
    }


def _analyze(source: str, opt: int, seed: int) -> dict:
    try:
        func = compile_to_graph(source, FILENAME)
    except CompileError as e:
        return {"ok": False, "error": e.render()}
    inputs = ref.random_inputs(func, np.random.default_rng(seed))
    try:
        want = ref.evaluate(func, inputs)
        oracle_error = None
    except RangeError as e:
        want, oracle_error = None, f"the program overflows i32 on these inputs: {e}"
    return {
        "ok": True,
        "opt": opt,
        "seed": seed,
        "inputs": {k: v.tolist() for k, v in inputs.items()},
        "outputNames": [v.name for v in func.outputs],
        "oracle": None if want is None else [w.tolist() for w in want],
        "oracleError": oracle_error,
        "gpu": _gpu(source, opt, inputs, want),
        "tpu": _tpu(source, opt),
    }


def analyze(source: str, opt: int = 1, seed: int = 42) -> str:
    """JSON string: compiled artifacts for both chips, the oracle, and the tiny-gpu measurement."""
    return json.dumps(
        _analyze(source, opt, seed),
        default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o),
    )
