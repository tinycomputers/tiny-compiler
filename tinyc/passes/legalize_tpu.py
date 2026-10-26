"""tiny-tpu legality: every matmul operand must be an i8 graph input or constant.

The array multiplies int8 x int8 only, there is no requantization yet, and the hardware silently
truncates wider operands, so anything else is rejected here rather than miscomputed.
"""

from tinyc.ir.errors import PassError


class LegalizeTpu:
    name = "legalize-tpu"

    def run(self, func, ctx):
        producer = {op.result.name: op for op in func.ops}
        for op in func.ops:
            if op.kind != "matmul":
                continue
            for v in op.operands[:2]:
                if v.type.dtype != "i8":
                    raise PassError(
                        f"tiny-tpu needs i8 matmul operands, but %{v.name} is {v.type} "
                        "(requantization is not supported yet)",
                        op.loc,
                    )
                made_by = producer.get(v.name)
                if made_by is not None and made_by.kind != "const":
                    raise PassError(
                        f"tiny-tpu matmul operands must be i8 inputs or constants, but "
                        f"%{v.name} is computed by '{made_by.kind}'",
                        op.loc,
                    )
