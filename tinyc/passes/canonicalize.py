"""Normal forms that make later passes simpler and let CSE see through operand order.

* `add`: the rank-2 operand comes first in a broadcast add; same-shape adds order their operands
  by definition order, so `a + b` and `b + a` become the same op.
* `relu(relu(x))` -> `relu(x)` (relu is idempotent). The dead outer op is left for DCE.
"""


class Canonicalize:
    name = "canonicalize"

    def run(self, func, ctx):
        position = {v.name: i for i, v in enumerate(func.args)}
        for i, op in enumerate(func.ops, start=len(func.args)):
            position[op.result.name] = i
        defs = {op.result.name: op for op in func.ops}

        for op in func.ops:
            if op.kind == "add":
                a, b = op.operands
                if a.type.rank < b.type.rank or (
                    a.type.shape == b.type.shape and position[a.name] > position[b.name]
                ):
                    op.operands = [b, a]
            elif op.kind == "relu":
                inner = defs.get(op.operands[0].name)
                if inner is not None and inner.kind == "relu":
                    func.replace_uses(op.result, op.operands[0])
