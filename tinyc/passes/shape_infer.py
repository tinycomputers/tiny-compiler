"""Recompute every result type from its operands (fixes IR that was parsed or hand-edited)."""

from tinyc.ir.errors import OpTypeError, PassError
from tinyc.ir.types import infer_result_type


class ShapeInfer:
    name = "shape-infer"

    def run(self, func, ctx):
        for op in func.ops:
            if op.kind == "const":
                continue
            try:
                op.result.type = infer_result_type(op.kind, [v.type for v in op.operands], op.attrs)
            except OpTypeError as e:
                raise PassError(str(e), op.loc) from None
