"""Linear-scan register allocation for a flattened kernel.

Pool: every register except the reserved R5 (block id), R6 (thread in block) and R7 (global tid),
which the simulator sets and the kernel only reads. There is no spilling: running out of
registers is a compile error that says how many were needed.

Liveness is a [first mention, last mention] interval per virtual register, extended around loops:
a value that is live when a loop starts and used inside it must survive until the loop's back
edge, because the next iteration reads it again. A register is reusable from the instruction at
which its previous owner is last read (an instruction reads its sources before writing).
"""

from __future__ import annotations

from dataclasses import dataclass

from tinyc.gpu.kernel_ir import RESERVED_REGS, SPECIAL_REGS, Special, VInst, VReg


class RegisterPressure(Exception):
    def __init__(self, message: str, loc=None):
        super().__init__(message)
        self.message, self.loc = message, loc


@dataclass
class Allocation:
    assignment: dict[VReg, int]
    registers_used: int  # distinct physical registers
    max_live: int  # peak number of simultaneously live values


def allocatable(num_registers: int) -> list[int]:
    if num_registers < 8:
        raise ValueError("tiny-gpu needs at least 8 registers (R7 holds the global thread id)")
    return [r for r in range(num_registers) if r not in RESERVED_REGS]


def live_intervals(insts: list[VInst]) -> dict[VReg, list[int]]:
    spans: dict[VReg, list[int]] = {}

    def touch(v, i):
        if isinstance(v, VReg):
            span = spans.setdefault(v, [i, i])
            span[0], span[1] = min(span[0], i), max(span[1], i)

    for i, inst in enumerate(insts):
        for s in inst.srcs:
            touch(s, i)
        touch(inst.dst, i)

    label_at = {inst.label: i for i, inst in enumerate(insts) if inst.op == "LABEL"}
    loops = [
        (label_at[inst.label], i)
        for i, inst in enumerate(insts)
        if inst.op in ("BRLT", "BRGT") and label_at[inst.label] <= i  # a backward branch
    ]
    changed = True
    while changed:  # nested loops: extending for the inner one can matter for the outer one
        changed = False
        for top, bottom in loops:
            for span in spans.values():
                if span[0] < top <= span[1] < bottom:
                    span[1] = bottom
                    changed = True
    return spans


def allocate(insts: list[VInst], num_registers: int, kernel: str = "?") -> Allocation:
    pool = allocatable(num_registers)
    spans = live_intervals(insts)
    order = sorted(spans, key=lambda v: (spans[v][0], spans[v][1], v.id))
    free = list(pool)
    active: list[VReg] = []
    assignment: dict[VReg, int] = {}
    max_live = 0
    for v in order:
        start = spans[v][0]
        for u in [u for u in active if spans[u][1] <= start]:  # last read at or before `start`
            active.remove(u)
            free.append(assignment[u])
        free.sort()
        if not free:
            at = next((i.loc for i in insts if i.dst == v), None)
            raise RegisterPressure(
                f"kernel @{kernel} needs more than {len(pool)} registers: {len(active) + 1} "
                f"values are live at once (R5-R7 are reserved by tiny-gpu); "
                f"raise --gpu-regs (now {num_registers})",
                at,
            )
        assignment[v] = free.pop(0)
        active.append(v)
        max_live = max(max_live, len(active))
    assert_reserved_untouched(insts, assignment)
    return Allocation(assignment, len(set(assignment.values())), max_live)


def assert_reserved_untouched(insts: list[VInst], assignment: dict[VReg, int]) -> None:
    """R5-R7 are only ever read: no virtual register may land on them, and no instruction writes one."""
    bad = [v for v, r in assignment.items() if r in RESERVED_REGS]
    assert not bad, f"allocator assigned reserved registers to {bad}"
    for inst in insts:
        if inst.dst is not None:
            assert assignment[inst.dst] not in RESERVED_REGS, f"{inst} writes a reserved register"


def reg_name(x, assignment: dict[VReg, int]) -> str:
    if isinstance(x, VReg):
        return f"R{assignment[x]}"
    if isinstance(x, Special):
        return f"R{SPECIAL_REGS[x.name]}"
    return str(x)
