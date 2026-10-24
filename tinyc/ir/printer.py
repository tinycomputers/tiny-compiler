"""Textual Graph IR (the format `ir.parser` reads back; see docs/IR.md)."""

from __future__ import annotations

from tinyc.ir.graph import Func, Op


def _op_text(op: Op) -> str:
    if op.kind == "const":
        assert op.data is not None
        body = f"const {op.data.tolist()!r}"
    else:
        body = f"{op.kind} " + ", ".join(f"%{v.name}" for v in op.operands)
    return f"%{op.result.name} = {body} : {op.result.type}"


def print_func(func: Func, *, locs: bool = False) -> str:
    args = ", ".join(f"%{a.name}: {a.type}" for a in func.args)
    rets = ", ".join(str(v.type) for v in func.outputs)
    lines = [f"func @{func.name}({args}) -> ({rets}) {{"]
    texts = [_op_text(op) for op in func.ops]
    width = max(map(len, texts), default=0)
    for op, text in zip(func.ops, texts):
        line = f"  {text}"
        if locs and op.loc is not None:
            line = f"{line.ljust(width + 2)}  {op.loc}"
        lines.append(line)
    lines.append("  return " + ", ".join(f"%{v.name}" for v in func.outputs))
    lines.append("}")
    return "\n".join(lines) + "\n"
