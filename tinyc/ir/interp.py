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


def _fits(v: int) -> bool:
    return I32[0] <= v <= I32[1]


def _rows(x):  # rank-1 -> a single row
    return x if isinstance(x[0], list) else [x]


def run(func: Func, inputs: Mapping[str, object]) -> list[np.ndarray]:
    env = {k: v.tolist() for k, v in validate_inputs(func, inputs).items()}
    for op in func.ops:
        name = op.result.name
        if op.kind == "const":
            r = op.data.tolist()
        elif op.kind == "relu":
            x = env[op.operands[0].name]
            r = (
                [[max(v, 0) for v in row] for row in x]
                if isinstance(x[0], list)
                else [max(v, 0) for v in x]
            )
        elif op.kind == "matmul":
            a, b = (env[v.name] for v in op.operands)
            r = [
                [sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))]
                for i in range(len(a))
            ]
        else:  # add, with row broadcast of a rank-1 operand
            a, b = (env[v.name] for v in op.operands)
            if isinstance(a[0], list) and not isinstance(b[0], list):
                r = [[x + y for x, y in zip(row, b)] for row in a]
            elif isinstance(b[0], list) and not isinstance(a[0], list):
                r = [[x + y for x, y in zip(a, row)] for row in b]
            elif isinstance(a[0], list):
                r = [[x + y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]
            else:
                r = [x + y for x, y in zip(a, b)]
        if not all(_fits(v) for row in _rows(r) for v in row):
            raise RangeError(f"%{name} ({op.kind}) does not fit in i32")
        env[name] = r
    return [np.array(env[v.name], dtype=np.int64) for v in func.outputs]
