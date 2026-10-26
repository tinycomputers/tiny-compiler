"""The oracle: numpy int64 evaluation of a Graph IR function.

Semantics (shared with `ir.interp`, which is an independent scalar implementation):
  * inputs must fit their declared dtype; every op computes in int64 on widened operands;
  * every op result must fit in i32, otherwise RangeError (tiny-gpu raises on overflow);
  * a fused matmul (`epilogue=bias,relu`) checks each stage's result, exactly like the unfused ops.
    Partial sums are not checked here; `passes.range_check` bounds them statically.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from tinyc.ir.errors import InputError, RangeError
from tinyc.ir.graph import Func
from tinyc.ir.types import RANGES, epilogue

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


def _check(r: np.ndarray, name: str, what: str) -> np.ndarray:
    if r.size and (r.min() < I32[0] or r.max() > I32[1]):
        raise RangeError(f"%{name} ({what}) does not fit in i32")
    return r


def eval_op(kind, xs, attrs=None, data=None, name="?") -> np.ndarray:
    """Evaluate one op on int64 arrays (used by `evaluate` and by constant folding)."""
    if kind == "const":
        return _check(np.array(data, dtype=np.int64), name, kind)
    if kind == "matmul":
        r = _check(xs[0] @ xs[1], name, "matmul")
        for stage in epilogue(attrs):
            r = _check(r + xs[2], name, "matmul bias add") if stage == "bias" else np.maximum(r, 0)
        return r
    if kind == "add":
        return _check(xs[0] + xs[1], name, "add")
    return np.maximum(xs[0], 0)  # relu


def evaluate(func: Func, inputs: Mapping[str, object]) -> list[np.ndarray]:
    env = dict(validate_inputs(func, inputs))
    for op in func.ops:
        xs = [env[v.name] for v in op.operands]
        env[op.result.name] = eval_op(op.kind, xs, op.attrs, op.data, op.result.name)
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


def seeded_expectations(func: Func, n: int, seed: int) -> list[tuple[dict, list[np.ndarray]]]:
    """(inputs, oracle outputs) for n seeded random inputs, skipping inputs that overflow i32 in
    the source program itself. Used by every `verify_each` translation-validation step."""
    out = []
    for i in range(n):
        inputs = random_inputs(func, np.random.default_rng(seed + i))
        try:
            out.append((inputs, evaluate(func, inputs)))
        except RangeError:
            continue
    return out
