"""Fuse `matmul` with the ops that follow it into one matmul with an epilogue.

    matmul -> add(bias[N]) -> relu     becomes     matmul{epilogue=bias,relu}

A stage is absorbed only when the previous result has exactly one use and is not an output.
The fused op sits where the last absorbed op was (so a late-defined bias is still defined before
use), keeps that op's result value, and takes its source location (the outermost expression).
"""

from tinyc.ir.graph import Op
from tinyc.ir.types import epilogue


class Fuse:
    name = "fuse"

    def run(self, func, ctx):
        users = func.users()
        out_names = {v.name for v in func.outputs}

        def sole_user(v):
            us = users.get(v.name, [])
            return us[0] if len(us) == 1 and v.name not in out_names else None

        skip: set[int] = set()
        fused_at: dict[int, Op] = {}
        for op in func.ops:
            if op.kind != "matmul" or epilogue(op.attrs) or id(op) in skip:
                continue
            cur, last, stages, operands = op.result, op, [], list(op.operands)
            nxt = sole_user(cur)
            if nxt is not None and nxt.kind == "add":
                other = nxt.operands[1] if nxt.operands[0] is cur else nxt.operands[0]
                if other.type.rank == 1 and other.type.shape[0] == cur.type.shape[1]:
                    stages.append("bias")
                    operands.append(other)
                    skip.add(id(last))
                    cur, last = nxt.result, nxt
                    nxt = sole_user(cur)
            if nxt is not None and nxt.kind == "relu":
                stages.append("relu")
                skip.add(id(last))
                cur, last = nxt.result, nxt
            if stages:
                fused_at[id(last)] = Op(
                    "matmul", operands, last.result, last.loc, None, {"epilogue": ",".join(stages)}
                )
        func.ops = [fused_at.get(id(o), o) for o in func.ops if id(o) not in skip]
