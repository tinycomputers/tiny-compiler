"""Assign tiny-gpu global-memory word addresses to every Graph IR value.

Inputs and constants are pinned (constants are the memory's initial contents); outputs are pinned
until the end. Every other intermediate is allocated when its op runs and its words are reused
once its last reader has finished. `i8` values take one int32 word each.

Reuse is only ever from ops *strictly before* the one allocating: an op's result never overlaps
the buffers that same op reads (its threads read and write concurrently).
"""

from __future__ import annotations

from dataclasses import dataclass

from tinyc.ir.graph import Func


@dataclass
class Region:
    name: str
    addr: int
    size: int
    dtype: str
    shape: tuple[int, ...]
    kind: str  # "input" | "const" | "temp"


@dataclass
class MemPlan:
    buffers: dict[str, Region]
    mem_size: int

    def addr(self, name: str) -> int:
        return self.buffers[name].addr


class _FreeList:
    """First-fit free list over [0, top) with coalescing."""

    def __init__(self):
        self.free: list[tuple[int, int]] = []  # (addr, size), sorted, non-adjacent
        self.top = 0

    def alloc(self, size: int) -> int:
        for i, (a, n) in enumerate(self.free):
            if n >= size:
                if n == size:
                    del self.free[i]
                else:
                    self.free[i] = (a + size, n - size)
                return a
        addr = self.top
        self.top += size
        return addr

    def release(self, addr: int, size: int) -> None:
        self.free.append((addr, size))
        self.free.sort()
        merged: list[tuple[int, int]] = []
        for a, n in self.free:
            if merged and merged[-1][0] + merged[-1][1] == a:
                merged[-1] = (merged[-1][0], merged[-1][1] + n)
            else:
                merged.append((a, n))
        self.free = merged


def plan_memory(func: Func) -> MemPlan:
    last_use: dict[str, int] = {}
    for i, op in enumerate(func.ops):
        for v in op.operands:
            last_use[v.name] = i
    pinned_until_end = {v.name for v in func.outputs}

    fl = _FreeList()
    buffers: dict[str, Region] = {}

    def place(v, kind: str) -> Region:
        r = Region(v.name, fl.alloc(v.type.size), v.type.size, v.type.dtype, v.type.shape, kind)
        buffers[v.name] = r
        return r

    for a in func.args:
        place(a, "input")
    for op in func.ops:
        if op.kind == "const":
            place(op.result, "const")

    live: dict[str, Region] = {}
    for i, op in enumerate(func.ops):
        if op.kind == "const":
            continue
        for name, r in list(live.items()):  # readers that finished before op i
            if last_use.get(name, -1) < i and name not in pinned_until_end:
                fl.release(r.addr, r.size)
                del live[name]
        live[op.result.name] = place(op.result, "temp")
    return MemPlan(buffers, fl.top)


def check_plan(func: Func, plan: MemPlan) -> None:
    """No two buffers that are live at the same time may overlap (used by tests and verify)."""
    n = len(func.ops)
    first = {a.name: -1 for a in func.args}
    last = {a.name: n for a in func.args}
    for i, op in enumerate(func.ops):
        first[op.result.name] = -1 if op.kind == "const" else i
        last[op.result.name] = n if op.kind == "const" else i
        for v in op.operands:
            last[v.name] = max(last[v.name], i)
    for v in func.outputs:
        last[v.name] = n
    names = list(plan.buffers)
    for x in range(len(names)):
        for y in range(x + 1, len(names)):
            a, b = plan.buffers[names[x]], plan.buffers[names[y]]
            alive_together = first[a.name] <= last[b.name] and first[b.name] <= last[a.name]
            overlap = a.addr < b.addr + b.size and b.addr < a.addr + a.size
            if alive_together and overlap:
                raise AssertionError(f"buffers {a.name} and {b.name} overlap while both live")
