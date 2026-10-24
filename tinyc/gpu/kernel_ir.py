"""Kernel IR: one tiny-gpu kernel as a structured program over virtual registers.

Every thread runs the same body. The launch grid is (blocks, threads per block); the simulator
gives each thread three read-only registers, exposed here as `Special` operands:
    block  = R5 (block id)    thread = R6 (thread in block)    tid = R7 (global thread id)
Virtual registers (`VReg`) are mapped to R0-R4 and R8.. by the register allocator; R5-R7 are
never written. tiny-gpu has no SUB/DIV/shift/MAX, so the IR has only what the ISA can do:
`set`/`add`/`mul`, global `ld`/`st`, a counted `for` (do-while, trip count >= 1) and `if le`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tinyc.ir.loc import Loc
from tinyc.ir.types import RANGES

SPECIAL_REGS = {"block": 5, "thread": 6, "tid": 7}
RESERVED_REGS = tuple(sorted(SPECIAL_REGS.values()))
I32_LO, I32_HI = RANGES["i32"]
MAX_THREADS = 1024


@dataclass(frozen=True)
class VReg:
    id: int

    def __str__(self) -> str:
        return f"v{self.id}"


@dataclass(frozen=True)
class Special:
    name: str  # "block" | "thread" | "tid"

    def __str__(self) -> str:
        return self.name


Operand = VReg | Special | int


@dataclass
class Assign:
    op: str  # "set" | "add" | "mul"
    dst: VReg
    args: tuple[Operand, ...]
    loc: Loc | None = None


@dataclass
class Load:
    dst: VReg
    addr: Operand
    loc: Loc | None = None


@dataclass
class Store:
    addr: Operand
    src: Operand
    loc: Loc | None = None


@dataclass
class For:
    """`var` counts start, start+1, ... while < stop. Always runs at least once (stop > start)."""

    var: VReg
    start: int
    stop: int
    body: list = field(default_factory=list)
    loc: Loc | None = None


@dataclass
class If:
    """Run `body` when `a <= b` (the only condition tiny-gpu's CMP + BRGT gives us for free)."""

    a: Operand
    b: Operand
    body: list = field(default_factory=list)
    loc: Loc | None = None


Stmt = Assign | Load | Store | For | If


@dataclass
class Kernel:
    name: str
    grid: tuple[int, int]  # (blocks, threads per block)
    body: list[Stmt]
    value: str  # name of the Graph IR value this kernel computes
    kind: str  # the Graph op: "matmul" | "add" | "relu"
    loc: Loc | None = None

    @property
    def threads(self) -> int:
        return self.grid[0] * self.grid[1]


class KernelBuilder:
    def __init__(self):
        self._next = 0

    def fresh(self) -> VReg:
        v = VReg(self._next)
        self._next += 1
        return v


class KernelError(Exception):
    pass


# --- verifier ---------------------------------------------------------------------------------


def verify_kernel(k: Kernel) -> None:
    problems: list[str] = []
    blocks, tpb = k.grid
    if blocks < 1 or tpb < 1:
        problems.append(f"grid {k.grid} must be positive")
    if blocks * tpb > MAX_THREADS:
        problems.append(f"{blocks * tpb} threads exceed the {MAX_THREADS}-thread launch limit")

    def operand(x: Operand, defined: set[VReg], where: str) -> None:
        if isinstance(x, VReg):
            if x not in defined:
                problems.append(f"{where}: {x} used before it is assigned")
        elif isinstance(x, Special):
            if x.name not in SPECIAL_REGS:
                problems.append(f"{where}: unknown special register '{x.name}'")
        elif isinstance(x, int) and not isinstance(x, bool):
            if not I32_LO <= x <= I32_HI:
                problems.append(f"{where}: immediate {x} does not fit in i32")
        else:
            problems.append(f"{where}: bad operand {x!r}")

    def walk(body: list, defined: set[VReg]) -> None:
        for s in body:
            where = type(s).__name__.lower()
            if isinstance(s, Assign):
                arity = {"set": 1, "add": 2, "mul": 2}.get(s.op)
                if arity is None or len(s.args) != arity:
                    problems.append(f"{where}: '{s.op}' with {len(s.args)} operands")
                for a in s.args:
                    operand(a, defined, s.op)
                defined.add(s.dst)
            elif isinstance(s, Load):
                operand(s.addr, defined, "ld")
                defined.add(s.dst)
            elif isinstance(s, Store):
                operand(s.addr, defined, "st")
                operand(s.src, defined, "st")
            elif isinstance(s, For):
                if s.stop <= s.start:
                    problems.append(f"for {s.var}: needs stop > start (got {s.start}..{s.stop})")
                defined.add(s.var)
                walk(s.body, defined)  # runs at least once, so its assignments are definite
            elif isinstance(s, If):
                operand(s.a, defined, "if")
                operand(s.b, defined, "if")
                walk(s.body, set(defined))  # conditional: its assignments are not definite after
            else:
                problems.append(f"unknown statement {s!r}")

    walk(k.body, set())
    if k.body and isinstance(k.body[-1], If):
        # tiny-gpu: branching to the end of the program costs a stray extra cycle (measured)
        problems.append("a kernel must not end with an `if` (its skip label would be at the end)")
    if problems:
        raise KernelError(f"invalid kernel @{k.name}:\n  " + "\n  ".join(problems))


# --- printing ---------------------------------------------------------------------------------


def _ops(args) -> str:
    return ", ".join(str(a) for a in args)


def _print_body(body: list, indent: int, out: list[str]) -> None:
    pad = "  " * indent
    for s in body:
        if isinstance(s, Assign):
            out.append(f"{pad}{s.dst} = {s.op} {_ops(s.args)}")
        elif isinstance(s, Load):
            out.append(f"{pad}{s.dst} = ld [{s.addr}]")
        elif isinstance(s, Store):
            out.append(f"{pad}st [{s.addr}], {s.src}")
        elif isinstance(s, For):
            out.append(f"{pad}for {s.var} in {s.start}..{s.stop} {{")
            _print_body(s.body, indent + 1, out)
            out.append(f"{pad}}}")
        else:
            out.append(f"{pad}if le {s.a}, {s.b} {{")
            _print_body(s.body, indent + 1, out)
            out.append(f"{pad}}}")


def print_kernel(k: Kernel) -> str:
    where = f" at {k.loc.line}:{k.loc.col}" if k.loc else ""
    head = (
        f"kernel @{k.name} grid=({k.grid[0]} blocks, {k.grid[1]} threads)"
        f"  // %{k.value} = {k.kind}{where}"
    )
    out = [head + " {"]
    _print_body(k.body, 1, out)
    out.append("}")
    return "\n".join(out) + "\n"


# --- flattening to virtual tiny-gpu instructions -----------------------------------------------


@dataclass
class VInst:
    """One tiny-gpu instruction over virtual registers (or a LABEL pseudo-instruction)."""

    op: str  # SET ADD MUL LD ST CMP BRLT BRGT LABEL
    dst: VReg | None = None
    srcs: tuple[Operand, ...] = ()
    label: str | None = None  # LABEL: the name defined; BR*: the target
    loc: Loc | None = None


def flatten(k: Kernel) -> list[VInst]:
    """Lower structured control flow to labels and branches, in tiny-gpu's own idiom:

    for:  SET var, start / L: body / ADD var, var, 1 / CMP var, stop / BRLT L
    if :  CMP a, b / BRGT skip / body / skip:
    """
    out: list[VInst] = []
    labels = iter(range(10**6))
    # Labels must not start with R/r: tiny-gpu's assembler would parse them as registers.

    def walk(body: list) -> None:
        for s in body:
            if isinstance(s, Assign):
                out.append(VInst(s.op.upper(), s.dst, s.args, loc=s.loc))
            elif isinstance(s, Load):
                out.append(VInst("LD", s.dst, (s.addr,), loc=s.loc))
            elif isinstance(s, Store):
                out.append(VInst("ST", None, (s.addr, s.src), loc=s.loc))
            elif isinstance(s, For):
                top = f"L{next(labels)}"
                out.append(VInst("SET", s.var, (s.start,), loc=s.loc))
                out.append(VInst("LABEL", label=top, loc=s.loc))
                walk(s.body)
                out.append(VInst("ADD", s.var, (s.var, 1), loc=s.loc))
                out.append(VInst("CMP", None, (s.var, s.stop), loc=s.loc))
                out.append(VInst("BRLT", label=top, loc=s.loc))
            else:
                skip = f"L{next(labels)}"
                out.append(VInst("CMP", None, (s.a, s.b), loc=s.loc))
                out.append(VInst("BRGT", label=skip, loc=s.loc))
                walk(s.body)
                out.append(VInst("LABEL", label=skip, loc=s.loc))

    walk(k.body)
    return out
