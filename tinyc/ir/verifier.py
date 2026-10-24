"""Structural and type checks for the Graph IR. Reports every problem, not just the first."""

from __future__ import annotations

import numpy as np

from tinyc.ir.errors import OpTypeError, VerifyError
from tinyc.ir.graph import Func
from tinyc.ir.types import OP_ARITY, RANGES, expected_arity, infer_result_type, type_problem


def verify(func: Func) -> None:
    problems: list[str] = []
    defined: dict[str, object] = {}

    def check_type(what: str, t) -> None:
        if (p := type_problem(t)) is not None:
            problems.append(f"{what}: {p}")

    for a in func.args:
        if a.name in defined:
            problems.append(f"%{a.name}: defined twice")
        defined[a.name] = a
        check_type(f"%{a.name}", a.type)

    for op in func.ops:
        r = op.result
        where = f"%{r.name} ({op.kind})"
        if op.kind not in OP_ARITY:
            problems.append(f"{where}: unknown op")
            continue
        try:
            arity = expected_arity(op.kind, op.attrs)
        except OpTypeError as e:
            problems.append(f"{where}: {e}")
            continue
        if op.attrs and (op.kind != "matmul" or set(op.attrs) != {"epilogue"}):
            problems.append(f"{where}: unsupported attributes {sorted(op.attrs)}")
            continue
        if len(op.operands) != arity:
            problems.append(f"{where}: expects {arity} operands, got {len(op.operands)}")
            continue
        if r.name in defined:
            problems.append(f"{where}: name already defined")
        check_type(where, r.type)
        operands_ok = True
        for v in op.operands:
            if defined.get(v.name) is not v:
                problems.append(f"{where}: operand %{v.name} is not defined before use")
                operands_ok = False
        if op.kind == "const":
            d = op.data
            if not isinstance(d, np.ndarray):
                problems.append(f"{where}: const has no data")
            elif d.shape != r.type.shape:
                problems.append(f"{where}: data has shape {list(d.shape)}, type says {r.type}")
            else:
                lo, hi = RANGES.get(r.type.dtype, (0, -1))
                if d.size and (d.min() < lo or d.max() > hi):
                    problems.append(f"{where}: data does not fit in {r.type.dtype}")
        else:
            if op.data is not None:
                problems.append(f"{where}: only const may carry data")
            if operands_ok:
                try:
                    want = infer_result_type(op.kind, [v.type for v in op.operands], op.attrs)
                except OpTypeError as e:
                    problems.append(f"{where}: {e}")
                else:
                    if want != r.type:
                        problems.append(
                            f"{where}: result type is {r.type} but operands give {want}"
                        )
        defined[r.name] = r

    if not func.outputs:
        problems.append("function has no outputs")
    for v in func.outputs:
        if defined.get(v.name) is not v:
            problems.append(f"return: %{v.name} is not defined")

    if problems:
        raise VerifyError(problems)
