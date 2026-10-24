"""Evaluate ops whose operands are all constants, using the oracle's own `eval_op`."""

from tinyc.ir import ref
from tinyc.ir.errors import PassError, RangeError


class ConstFold:
    name = "const-fold"

    def run(self, func, ctx):
        consts = {op.result.name: op.data for op in func.ops if op.kind == "const"}
        for op in func.ops:
            if op.kind == "const" or not all(v.name in consts for v in op.operands):
                continue
            xs = [consts[v.name] for v in op.operands]
            try:
                data = ref.eval_op(op.kind, xs, op.attrs, op.data, op.result.name)
            except RangeError as e:
                raise PassError(f"constant expression overflows i32 ({e})", op.loc) from None
            op.kind, op.operands, op.attrs, op.data = "const", [], {}, data
            consts[op.result.name] = data
