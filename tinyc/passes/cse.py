"""Common-subexpression elimination: identical ops (and identical constants) share one result."""


def _key(op):
    data = None if op.data is None else (op.data.shape, op.data.tobytes())
    return (
        op.kind,
        tuple(id(v) for v in op.operands),
        tuple(sorted(op.attrs.items())),
        str(op.result.type),
        data,
    )


class CSE:
    name = "cse"

    def run(self, func, ctx):
        seen, dead = {}, set()
        for op in func.ops:
            first = seen.setdefault(_key(op), op.result)
            if first is not op.result:
                func.replace_uses(op.result, first)  # later ops now hash on the survivor
                dead.add(id(op))
        func.ops = [op for op in func.ops if id(op) not in dead]
