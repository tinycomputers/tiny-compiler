"""Tile IR: the schedule of one program on tiny-tpu.

A program is a list of commands over named buffers. `Pass` runs one 4x4 int8 tile product on
the chip (the only chip work); everything else (`Acc`, `Epilogue`, `HostAdd`, `HostRelu`,
`Alloc`) is *host* work and is labelled as such everywhere (PLAN §2.2: the chip always starts a
pass from psum = 0, so accumulating over K tiles, bias and relu happen on the host).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TILE = 4
# Measured: TinyTpuSim.getCycleCount() after one pass (14 from start to done + 1 for the last
# element to latch). Pinned by tests/test_targets_contract.py.
CYCLES_PER_PASS = 15
CAPACITY_MACS_PER_PASS = TILE * TILE * TILE


@dataclass(frozen=True)
class TileRef:
    """A 4x4 window of `buf` at element offset (r, c); out-of-bounds elements read as 0."""

    buf: str
    r: int
    c: int


@dataclass
class Buffer:
    name: str
    dtype: str
    shape: tuple[int, ...]
    kind: str  # "input" | "const" | "temp"
    data: np.ndarray | None = None  # const only


@dataclass
class Alloc:
    buf: str


@dataclass
class Pass:
    id: int
    a: TileRef
    b: TileRef


@dataclass
class Acc:
    """dst window += result of pass `pass_id`; then the optional epilogue (bias, relu)."""

    pass_id: int
    dst: TileRef
    bias: str | None = None
    relu: bool = False


@dataclass
class Epilogue:
    """Apply bias and/or relu to a dst window with no pass feeding it."""

    dst: TileRef
    bias: str | None = None
    relu: bool = False


@dataclass
class HostAdd:
    dst: str
    a: str
    b: str


@dataclass
class HostRelu:
    dst: str
    src: str


Command = Alloc | Pass | Acc | Epilogue | HostAdd | HostRelu


@dataclass
class Schedule:
    name: str
    buffers: dict[str, Buffer]
    commands: list[Command]
    inputs: list[str]
    outputs: list[str]
    meta: dict = field(default_factory=dict)  # e.g. {"passes_skipped": 3}


class ScheduleError(Exception):
    pass


def verify_schedule(s: Schedule) -> None:
    """Structural checks. Raises ScheduleError listing every problem."""
    problems: list[str] = []
    bufs = s.buffers

    def buf(name, what, rank=None, dtype=None):
        b = bufs.get(name)
        if b is None:
            problems.append(f"{what}: unknown buffer '{name}'")
        elif rank is not None and len(b.shape) != rank:
            problems.append(f"{what}: '{name}' must have rank {rank}, has {len(b.shape)}")
        elif dtype is not None and b.dtype != dtype:
            problems.append(f"{what}: '{name}' must be {dtype}, is {b.dtype}")
        return b

    def window(ref, what):
        b = buf(ref.buf, what, rank=2)
        if (
            b is not None
            and len(b.shape) == 2
            and not (0 <= ref.r < b.shape[0] and 0 <= ref.c < b.shape[1])
        ):
            problems.append(f"{what}: window at ({ref.r},{ref.c}) lies outside '{ref.buf}'")
        return b

    for name in s.inputs + s.outputs:
        buf(name, "interface")
    seen_passes: dict[int, Pass] = {}
    consumed: set[int] = set()
    for i, cmd in enumerate(s.commands):
        where = f"command {i} ({type(cmd).__name__})"
        if isinstance(cmd, Alloc):
            buf(cmd.buf, where)
        elif isinstance(cmd, Pass):
            if cmd.id in seen_passes:
                problems.append(f"{where}: duplicate pass id {cmd.id}")
            seen_passes[cmd.id] = cmd
            a, b = window(cmd.a, where + " a"), window(cmd.b, where + " b")
            for x in (a, b):
                if x is not None and x.dtype != "i8":
                    problems.append(f"{where}: operand '{x.name}' must be i8 (chip is int8 x int8)")
            if a is not None and b is not None and cmd.a.c != cmd.b.r:
                problems.append(f"{where}: A columns start at {cmd.a.c} but B rows at {cmd.b.r}")
        elif isinstance(cmd, (Acc, Epilogue)):
            dst = window(cmd.dst, where)
            if dst is not None and dst.dtype != "i32":
                problems.append(f"{where}: destination '{dst.name}' must be i32")
            if cmd.bias is not None:
                buf(cmd.bias, where + " bias", rank=1)
            if isinstance(cmd, Acc):
                if cmd.pass_id not in seen_passes:
                    problems.append(f"{where}: pass #{cmd.pass_id} not run yet")
                elif cmd.pass_id in consumed:
                    problems.append(f"{where}: pass #{cmd.pass_id} accumulated twice")
                consumed.add(cmd.pass_id)
        elif isinstance(cmd, HostAdd):
            for n in (cmd.dst, cmd.a, cmd.b):
                buf(n, where)
        elif isinstance(cmd, HostRelu):
            for n in (cmd.dst, cmd.src):
                buf(n, where)
    for pid in seen_passes:
        if pid not in consumed:
            problems.append(f"pass #{pid} is never accumulated")
    if problems:
        raise ScheduleError("invalid tile schedule:\n  " + "\n  ".join(problems))


# --- printing ---------------------------------------------------------------------------------


def _win(ref: TileRef) -> str:
    return f"%{ref.buf}[{ref.r}:{ref.r + TILE},{ref.c}:{ref.c + TILE}]"


def _epi(bias, relu) -> str:
    parts = ([f"bias=%{bias}"] if bias else []) + (["relu"] if relu else [])
    return f" {{{', '.join(parts)}}}" if parts else ""


def print_schedule(s: Schedule) -> str:
    args = ", ".join(f"%{n}: {s.buffers[n].dtype}[{','.join(map(str, s.buffers[n].shape))}]"
                     for n in s.inputs)  # fmt: skip
    outs = ", ".join(f"%{n}" for n in s.outputs)
    lines = [f"tpu.schedule @{s.name}({args}) -> ({outs}) tile={TILE}x{TILE} {{"]
    for b in s.buffers.values():
        if b.kind == "const":
            t = f"{b.dtype}[{','.join(map(str, b.shape))}]"
            lines.append(f"  host.const %{b.name} : {t} = {b.data.tolist()!r}")
    for cmd in s.commands:
        if isinstance(cmd, Alloc):
            b = s.buffers[cmd.buf]
            lines.append(f"  host.alloc %{b.name} : {b.dtype}[{','.join(map(str, b.shape))}] = 0")
        elif isinstance(cmd, Pass):
            lines.append(
                f"  tpu.pass #{cmd.id} a={_win(cmd.a)} b={_win(cmd.b)}  // {CYCLES_PER_PASS} cyc"
            )
        elif isinstance(cmd, Acc):
            lines.append(f"  host.acc {_win(cmd.dst)} += #{cmd.pass_id}{_epi(cmd.bias, cmd.relu)}")
        elif isinstance(cmd, Epilogue):
            lines.append(f"  host.epilogue {_win(cmd.dst)}{_epi(cmd.bias, cmd.relu)}")
        elif isinstance(cmd, HostAdd):
            lines.append(f"  host.add %{cmd.dst} = %{cmd.a}, %{cmd.b}")
        else:
            lines.append(f"  host.relu %{cmd.dst} = %{cmd.src}")
    lines.append("}")
    return "\n".join(lines) + "\n"
