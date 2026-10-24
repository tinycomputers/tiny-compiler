"""Dead-code elimination: drop ops whose results never reach an output. Inputs are kept."""


class DCE:
    name = "dce"

    def run(self, func, ctx):
        live = {v.name for v in func.outputs}
        keep = []
        for op in reversed(func.ops):
            if op.result.name in live:
                keep.append(op)
                live.update(v.name for v in op.operands)
        func.ops = keep[::-1]
