"""Graph IR (legalized for tiny-tpu) -> Tile IR.

Each matmul becomes one 4x4 pass per (output tile, K tile), in the order tiny-tpu's own
`tilingSchedule()` uses: output tiles row-major, K innermost. After every pass the host
accumulates the tile into the result; the last accumulation of an output tile carries the fused
bias/relu epilogue. `add`/`relu` ops that were not fused are whole-buffer host ops.
"""

from __future__ import annotations

from tinyc.ir.graph import Func
from tinyc.ir.types import epilogue
from tinyc.tpu.tile_ir import (
    TILE,
    Acc,
    Alloc,
    Buffer,
    HostAdd,
    HostRelu,
    Pass,
    Schedule,
    TileRef,
)


def lower(func: Func) -> Schedule:
    s = Schedule(
        name=func.name,
        buffers={},
        commands=[],
        inputs=[a.name for a in func.args],
        outputs=[v.name for v in func.outputs],
    )
    for a in func.args:
        s.buffers[a.name] = Buffer(a.name, a.type.dtype, a.type.shape, "input")
    next_pass = 0
    for op in func.ops:
        r = op.result
        if op.kind == "const":
            s.buffers[r.name] = Buffer(r.name, r.type.dtype, r.type.shape, "const", op.data)
            continue
        s.buffers[r.name] = Buffer(r.name, r.type.dtype, r.type.shape, "temp")
        if op.kind == "matmul":
            a, b = op.operands[0], op.operands[1]
            stages = epilogue(op.attrs)
            bias = op.operands[2].name if "bias" in stages else None
            relu = "relu" in stages
            (m, k), n = a.type.shape, b.type.shape[1]
            s.commands.append(Alloc(r.name))
            for i0 in range(0, m, TILE):
                for j0 in range(0, n, TILE):
                    ks = list(range(0, k, TILE))
                    for idx, k0 in enumerate(ks):
                        last = idx == len(ks) - 1
                        s.commands.append(
                            Pass(next_pass, TileRef(a.name, i0, k0), TileRef(b.name, k0, j0))
                        )
                        s.commands.append(
                            Acc(
                                next_pass,
                                TileRef(r.name, i0, j0),
                                bias if last else None,
                                relu and last,
                            )
                        )
                        next_pass += 1
        elif op.kind == "add":
            s.commands.append(HostAdd(r.name, op.operands[0].name, op.operands[1].name))
        else:  # relu
            s.commands.append(HostRelu(r.name, op.operands[0].name))
    return s
