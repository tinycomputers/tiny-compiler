"""Static cost bounds for kernels, to be compared with what the simulator measures.

Loads, stores and launch counts are exact. Instruction and cycle counts are data dependent
exactly where a kernel has an `if` (relu): a thread that clamps runs one more instruction, and a
thread that does not clamp takes a skip branch, which ends its simulator cycle. The bounds are
the two extremes; `gpu.interp` gives the exact numbers for a given input.
"""

from __future__ import annotations

from dataclasses import dataclass

from tinyc.gpu.kernel_ir import Assign, For, Kernel, Load, Store


@dataclass
class _Cost:
    imin: int = 0
    imax: int = 0
    tmin: int = 0  # taken branches per thread
    tmax: int = 0
    loads: int = 0
    stores: int = 0

    def seq(self, o: _Cost) -> _Cost:
        return _Cost(self.imin + o.imin, self.imax + o.imax, self.tmin + o.tmin,
                     self.tmax + o.tmax, self.loads + o.loads, self.stores + o.stores)  # fmt: skip


def _cost(body: list) -> _Cost:
    total = _Cost()
    for s in body:
        if isinstance(s, (Assign, Load, Store)):
            total = total.seq(_Cost(1, 1, 0, 0, isinstance(s, Load), isinstance(s, Store)))
        elif isinstance(s, For):
            n, b = s.stop - s.start, _cost(s.body)
            total = total.seq(_Cost(
                1 + n * (b.imin + 3), 1 + n * (b.imax + 3),
                (n - 1) + n * b.tmin, (n - 1) + n * b.tmax, n * b.loads, n * b.stores,
            ))  # fmt: skip
        else:
            b = _cost(s.body)
            assert not (
                b.loads or b.stores
            ), "memory ops inside `if` would make loads data dependent"
            # skipped: 2 instructions + 1 taken branch; executed: 2 + body, body's own branches
            total = total.seq(_Cost(2, 2 + b.imax, min(1, b.tmin), max(1, b.tmax), 0, 0))
    return total


def kernel_cost(k: Kernel, registers: int) -> dict:
    c, t = _cost(k.body), k.threads
    return {
        "name": k.name,
        "grid": list(k.grid),
        "threads": t,
        "instructions": {"min": t * c.imin, "max": t * c.imax},
        "cycles": {"min": c.tmin + 1, "max": c.tmax + 1},
        "loads": t * c.loads,
        "stores": t * c.stores,
        "registers": registers,
    }


def estimate(kernels: list[Kernel], registers: dict[str, int]) -> dict:
    per = [kernel_cost(k, registers[k.name]) for k in kernels]
    return {
        "launches": len(per),
        "threads": sum(p["threads"] for p in per),
        "instructions": {
            "min": sum(p["instructions"]["min"] for p in per),
            "max": sum(p["instructions"]["max"] for p in per),
        },
        "cycles": {
            "min": sum(p["cycles"]["min"] for p in per),
            "max": sum(p["cycles"]["max"] for p in per),
        },
        "loads": sum(p["loads"] for p in per),
        "stores": sum(p["stores"] for p in per),
        "registers": max((p["registers"] for p in per), default=0),
        "per_kernel": per,
    }
