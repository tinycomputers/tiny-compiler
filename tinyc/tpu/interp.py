"""Reference interpreter for `.ttpu.json`, written against the format spec (numpy, no chip).

Independent of the JS executor (`runtime/ttpu-exec.mjs`): same spec, different language. Together
with the numpy oracle it triangulates the RTL run. The chip's pass is modelled as a plain tile
matmul; the RTL is the only place that actually checks the chip's behaviour.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from tinyc.ir.errors import InputError, RangeError
from tinyc.ir.types import RANGES

I32_LO, I32_HI = RANGES["i32"]


def _fit(a: np.ndarray, what: str) -> np.ndarray:
    if a.size and (a.min() < I32_LO or a.max() > I32_HI):
        raise RangeError(f"{what} does not fit in i32")
    return a


def _tile(buf: np.ndarray, ref: dict, t: int) -> np.ndarray:
    out = np.zeros((t, t), dtype=np.int64)
    sub = buf[ref["r"] : ref["r"] + t, ref["c"] : ref["c"] + t]
    out[: sub.shape[0], : sub.shape[1]] = sub
    return out


def _window(buf: np.ndarray, ref: dict, t: int):
    r, c = ref["r"], ref["c"]
    return buf[r : r + t, c : c + t]  # a view, clipped to the buffer


def _epilogue(view: np.ndarray, cmd: dict, env, ref: dict) -> None:
    if "bias" in cmd:
        bias = env[cmd["bias"]][ref["c"] : ref["c"] + view.shape[1]]
        view[:] = _fit(view + bias, "bias add")
    if cmd.get("relu"):
        view[:] = np.maximum(view, 0)


def run_program(program: dict, inputs: Mapping[str, object]) -> dict:
    """Execute; returns {"outputs": {name: int64 array}, "passes": n, "host_ops": {...}}."""
    t = program["tile"]
    env: dict[str, np.ndarray] = {}
    for name, b in program["buffers"].items():
        shape, kind = tuple(b["shape"]), b["kind"]
        if kind == "const":
            env[name] = np.array(b["data"], dtype=np.int64)
        elif kind == "input":
            if name not in inputs:
                raise InputError(f"missing input '{name}'")
            arr = np.asarray(inputs[name], dtype=np.int64)
            lo, hi = RANGES[b["dtype"]]
            if arr.shape != shape or (arr.size and (arr.min() < lo or arr.max() > hi)):
                raise InputError(f"input '{name}' must be {b['dtype']}{list(shape)}")
            env[name] = arr
        else:
            env[name] = np.zeros(shape, dtype=np.int64)

    tmp: dict[int, np.ndarray] = {}
    passes, host_ops = 0, {"alloc": 0, "acc": 0, "epilogue": 0, "host_add": 0, "host_relu": 0}
    for cmd in program["commands"]:
        op = cmd["op"]
        if op == "pass":
            a = _tile(env[cmd["a"]["buf"]], cmd["a"], t)
            b = _tile(env[cmd["b"]["buf"]], cmd["b"], t)
            tmp[cmd["id"]] = _fit(a @ b, "pass result")
            passes += 1
            continue
        host_ops[op] += 1
        if op == "alloc":
            env[cmd["buf"]][:] = 0
        elif op == "acc":
            view = _window(env[cmd["dst"]["buf"]], cmd["dst"], t)
            view[:] = _fit(view + tmp[cmd["pass"]][: view.shape[0], : view.shape[1]], "accumulate")
            _epilogue(view, cmd, env, cmd["dst"])
        elif op == "epilogue":
            _epilogue(_window(env[cmd["dst"]["buf"]], cmd["dst"], t), cmd, env, cmd["dst"])
        elif op == "host_add":
            env[cmd["dst"]][:] = _fit(env[cmd["a"]] + env[cmd["b"]], "add")
        else:  # host_relu
            env[cmd["dst"]][:] = np.maximum(env[cmd["src"]], 0)
    return {
        "outputs": {n: env[n] for n in program["outputs"]},
        "passes": passes,
        "host_ops": host_ops,
    }
