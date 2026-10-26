"""The oracle: numpy int64 evaluation of a Graph IR function.

Semantics (shared with `ir.interp`, which is an independent scalar implementation):
  * inputs must fit their declared dtype; every op computes in int64 on widened operands;
  * every op result must fit in i32, otherwise RangeError (tiny-gpu raises on overflow).
    Only final results are checked here; Phase 2's range analysis bounds partial sums as well.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from tinyc.ir.errors import InputError, RangeError
from tinyc.ir.graph import Func
from tinyc.ir.types import RANGES

I32 = RANGES["i32"]


def validate_inputs(func: Func, inputs: Mapping[str, object]) -> dict[str, np.ndarray]:
    """Check names, shapes and dtype ranges; return int64 arrays."""
    expected = {a.name for a in func.args}
    if extra := set(inputs) - expected:
        raise InputError(f"unexpected inputs: {sorted(extra)}")
    out = {}
    for a in func.args:
        if a.name not in inputs:
            raise InputError(f"missing input '{a.name}' ({a.type})")
        arr = np.asarray(inputs[a.name], dtype=np.int64)
        if arr.shape != a.type.shape:
            raise InputError(f"input '{a.name}' has shape {list(arr.shape)}, expected {a.type}")
        lo, hi = RANGES[a.type.dtype]
        if arr.size and (arr.min() < lo or arr.max() > hi):
            raise InputError(f"input '{a.name}' has values outside {a.type.dtype} ({lo}..{hi})")
        out[a.name] = arr
    return out


def evaluate(func: Func, inputs: Mapping[str, object]) -> list[np.ndarray]:
    env = {k: v for k, v in validate_inputs(func, inputs).items()}
    for op in func.ops:
        x = [env[v.name] for v in op.operands]
        if op.kind == "const":
            r = op.data
        elif op.kind == "matmul":
            r = x[0] @ x[1]
        elif op.kind == "add":
            r = x[0] + x[1]
        else:  # relu
            r = np.maximum(x[0], 0)
        if r.size and (r.min() < I32[0] or r.max() > I32[1]):
            raise RangeError(f"%{op.result.name} ({op.kind}) does not fit in i32")
        env[op.result.name] = r
    return [env[v.name] for v in func.outputs]


def random_inputs(
    func: Func, rng: np.random.Generator, i32_range: tuple[int, int] = (-1000, 1000)
) -> dict[str, np.ndarray]:
    """Seeded random inputs: full-range i8, small-range i32 (so i32 matmuls cannot overflow)."""
    out = {}
    for a in func.args:
        lo, hi = RANGES["i8"] if a.type.dtype == "i8" else i32_range
        out[a.name] = rng.integers(lo, hi + 1, size=a.type.shape, dtype=np.int64)
    return out
