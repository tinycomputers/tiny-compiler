"""Scalar Python interpreter for the Graph IR.

Deliberately independent of `ir.ref` (plain loops over Python ints, no numpy arithmetic) so the
two can be cross-checked. It is the template for the Tile IR / Kernel IR interpreters later.
Same semantics as `ref.evaluate`; see that module's docstring.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from tinyc.ir.errors import RangeError
from tinyc.ir.graph import Func
from tinyc.ir.ref import I32, validate_inputs
from tinyc.ir.types import epilogue


def _fits(v: int) -> bool:
    return I32[0] <= v <= I32[1]


def _rows(x):  # a rank-1 value is a single row
    return x if isinstance(x[0], list) else [x]


def _check(r, name: str, what: str):
    if not all(_fits(v) for row in _rows(r) for v in row):
        raise RangeError(f"%{name} ({what}) does not fit in i32")
    return r


def _matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))]
            for i in range(len(a))]  # fmt: skip


def _add(a, b):  # same shape, or a rank-1 operand broadcast over the rows of a rank-2 one
    if isinstance(a[0], list) and not isinstance(b[0], list):
        return [[x + y for x, y in zip(row, b)] for row in a]
    if isinstance(b[0], list) and not isinstance(a[0], list):
        return [[x + y for x, y in zip(a, row)] for row in b]
    if isinstance(a[0], list):
        return [[x + y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]
    return [x + y for x, y in zip(a, b)]


def _relu(x):
    if isinstance(x[0], list):
        return [[max(v, 0) for v in row] for row in x]
    return [max(v, 0) for v in x]


def run(func: Func, inputs: Mapping[str, object]) -> list[np.ndarray]:
    env = {k: v.tolist() for k, v in validate_inputs(func, inputs).items()}
    for op in func.ops:
        name, xs = op.result.name, [env[v.name] for v in op.operands]
        if op.kind == "const":
            r = op.data.tolist()
        elif op.kind == "relu":
            r = _relu(xs[0])
        elif op.kind == "matmul":
            r = _check(_matmul(xs[0], xs[1]), name, "matmul")
            for stage in epilogue(op.attrs):
                r = _check(_add(r, xs[2]), name, "matmul bias add") if stage == "bias" else _relu(r)
        else:
            r = _check(_add(xs[0], xs[1]), name, "add")
        env[name] = r
    return [np.array(env[v.name], dtype=np.int64) for v in func.outputs]
