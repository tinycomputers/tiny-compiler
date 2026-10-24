"""Static cost model for a tile schedule. It must equal what the RTL executor measures."""

from __future__ import annotations

from tinyc.tpu.tile_ir import (
    CAPACITY_MACS_PER_PASS,
    CYCLES_PER_PASS,
    TILE,
    Acc,
    Alloc,
    Epilogue,
    HostAdd,
    HostRelu,
    Pass,
    Schedule,
)

HOST_OP_KINDS = {Alloc: "alloc", Acc: "acc", Epilogue: "epilogue", HostAdd: "host_add",
                 HostRelu: "host_relu"}  # fmt: skip


def useful_macs(s: Schedule, p: Pass) -> int:
    """Multiply-adds in this pass that are not zero padding (rows x k x cols actually in range)."""
    (m, k), n = s.buffers[p.a.buf].shape, s.buffers[p.b.buf].shape[1]
    return (
        max(0, min(TILE, m - p.a.r)) * max(0, min(TILE, k - p.a.c)) * max(0, min(TILE, n - p.b.c))
    )


def estimate(s: Schedule) -> dict:
    passes = [c for c in s.commands if isinstance(c, Pass)]
    host_ops = {kind: 0 for kind in HOST_OP_KINDS.values()}
    for c in s.commands:
        if type(c) in HOST_OP_KINDS:
            host_ops[HOST_OP_KINDS[type(c)]] += 1
    useful = sum(useful_macs(s, p) for p in passes)
    capacity = len(passes) * CAPACITY_MACS_PER_PASS
    return {
        "passes": len(passes),
        "passes_skipped": s.meta.get("passes_skipped", 0),
        "cycles_per_pass": CYCLES_PER_PASS,
        "rtl_cycles": len(passes) * CYCLES_PER_PASS,
        "host_ops": host_ops,
        "useful_macs": useful,
        "capacity_macs": capacity,
        "utilization": round(useful / capacity, 4) if capacity else None,
    }
