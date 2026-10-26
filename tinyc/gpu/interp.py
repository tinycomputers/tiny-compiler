"""Reference interpreter for the Kernel IR, with the same dynamic counts the simulator reports.

It runs the *structured* kernels (before register allocation and assembly), thread by thread,
so it is independent of the lowering to tiny-gpu instructions and of the allocator. It also counts
what tiny-gpu would execute, so a test can require interpreter counts == simulator counts exactly:
that checks lowering, allocation and emission together.

Counting follows the idioms in `kernel_ir.flatten` and tiny-gpu's `step()`: a thread's cycle ends
at a taken branch, so a thread needs (taken branches + 1) cycles and a launch needs the maximum
over its threads.

Barriers (`SYNCB`) are modelled from tiny-gpu's source: a thread that reaches the barrier stops
for that step, and a waiting thread executes the `SYNCB` instruction *again on every following
step* until every active thread of its block is waiting; then the block is released and resumes
on the next step. So with arrival step s_t = (taken branches before the barrier) + 1 and release
step R = max s_t over the block, a thread executes `SYNCB` (R - s_t + 1) times and finishes at
step R + (taken branches after the barrier) + 1.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from tinyc.gpu.kernel_ir import (
    I32_HI,
    I32_LO,
    Assign,
    Barrier,
    For,
    Kernel,
    Load,
    SharedLoad,
    SharedStore,
    Special,
    Store,
)
from tinyc.ir.errors import InputError, RangeError


class GpuFault(Exception):
    """A thread touched memory outside its bounds (the simulator would silently misbehave)."""


@dataclass
class KernelCounts:
    name: str
    threads: int
    cycles: int  # max over threads of the step on which the thread finishes
    ops: Counter = field(default_factory=Counter)  # tiny-gpu op name -> executions, all threads

    @property
    def instructions(self) -> int:
        return sum(self.ops.values())


def _run_thread(
    kernel: Kernel, body: list, memory: list, shared: list, env: dict, ops: Counter
) -> int:
    """Execute `body` for one thread; returns its taken-branch count."""
    taken = 0

    def val(x):
        if isinstance(x, Special):
            return env[x.name]
        return x if isinstance(x, int) else env[x]

    def fit(v: int, what: str) -> int:
        if not I32_LO <= v <= I32_HI:
            raise RangeError(f"{what} overflows i32 in kernel @{kernel.name}")
        return v

    def addr(a: int, size: int, space: str) -> int:
        if not 0 <= a < size:
            raise GpuFault(f"kernel @{kernel.name}: {space} address {a} outside [0, {size})")
        return a

    def walk(stmts):
        nonlocal taken
        for s in stmts:
            if isinstance(s, Assign):
                ops[s.op.upper()] += 1
                a = [val(x) for x in s.args]
                v = a[0] if s.op == "set" else a[0] + a[1] if s.op == "add" else a[0] * a[1]
                env[s.dst] = fit(v, s.op)
            elif isinstance(s, Load):
                ops["LD"] += 1
                env[s.dst] = memory[addr(val(s.addr), len(memory), "global")]
            elif isinstance(s, Store):
                ops["ST"] += 1
                memory[addr(val(s.addr), len(memory), "global")] = fit(val(s.src), "st")
            elif isinstance(s, SharedLoad):
                ops["SHLD"] += 1
                env[s.dst] = shared[addr(val(s.saddr), len(shared), "shared")]
            elif isinstance(s, SharedStore):
                ops["SHST"] += 1
                shared[addr(val(s.saddr), len(shared), "shared")] = fit(val(s.src), "shst")
            elif isinstance(s, Barrier):
                raise TypeError("barriers are handled by run_block, at the top level only")
            elif isinstance(s, For):
                ops["SET"] += 1
                env[s.var] = s.start
                trips = s.stop - s.start
                for it in range(trips):
                    walk(s.body)
                    env[s.var] = env[s.var] + 1
                    ops["ADD"] += 1
                    ops["CMP"] += 1
                    ops["BRLT"] += 1
                    if it < trips - 1:
                        taken += 1  # the back edge is taken on every iteration but the last
            else:
                ops["CMP"] += 1
                ops["BRGT"] += 1
                if val(s.a) <= val(s.b):
                    walk(s.body)
                else:
                    taken += 1  # skip branch taken

    walk(body)
    return taken


def run_block(kernel: Kernel, memory: list, block: int, ops: Counter) -> int:
    """Run every thread of one block; returns the step on which the block's last thread finishes."""
    tpb = kernel.grid[1]
    shared = [0] * kernel.shared_size
    split = next((i for i, s in enumerate(kernel.body) if isinstance(s, Barrier)), None)
    before = kernel.body if split is None else kernel.body[:split]
    after = [] if split is None else kernel.body[split + 1 :]
    assert not any(isinstance(s, Barrier) for s in after), "at most one barrier per kernel"

    envs = [{"block": block, "thread": t, "tid": block * tpb + t} for t in range(tpb)]
    taken1 = [_run_thread(kernel, before, memory, shared, env, ops) for env in envs]
    if split is None:
        return max(taken1) + 1
    arrival = [t + 1 for t in taken1]
    release = max(arrival)
    for a in arrival:
        ops["SYNCB"] += release - a + 1  # waiting threads re-execute the barrier every step
    taken2 = [_run_thread(kernel, after, memory, shared, env, ops) for env in envs]
    return max(release + t + 1 for t in taken2)


def initial_memory(launch: Mapping, inputs: Mapping[str, object]) -> list[int]:
    """Memory image before the first kernel: constants and inputs placed, the rest zero."""
    memory = [0] * launch["mem_size"]
    for name, b in launch["buffers"].items():
        if b["kind"] == "const":
            flat = np.asarray(b["data"], dtype=np.int64).reshape(-1).tolist()
        elif b["kind"] == "input":
            if name not in inputs:
                raise InputError(f"missing input '{name}'")
            arr = np.asarray(inputs[name], dtype=np.int64)
            if list(arr.shape) != b["shape"]:
                raise InputError(f"input '{name}' must have shape {b['shape']}")
            flat = arr.reshape(-1).tolist()
        else:
            continue
        memory[b["addr"] : b["addr"] + len(flat)] = flat
    return memory


def read_outputs(launch: Mapping, memory) -> dict[str, np.ndarray]:
    out = {}
    for name in launch["outputs"]:
        b = launch["buffers"][name]
        size = int(np.prod(b["shape"]))
        out[name] = np.array(memory[b["addr"] : b["addr"] + size], dtype=np.int64).reshape(
            b["shape"]
        )
    return out


def run_program(prog, inputs: Mapping[str, object]) -> dict:
    """Run a tinyc.gpu.GpuProgram's kernels. Returns {"outputs": {...}, "kernels": [KernelCounts]}."""
    launch = prog.launch_core  # buffers and memory size only: no need for the cost estimate
    memory = initial_memory(launch, inputs)
    results = []
    for k in prog.kernels:
        ops: Counter = Counter()
        worst = max(run_block(k, memory, block, ops) for block in range(k.grid[0]))
        results.append(KernelCounts(k.name, k.threads, worst, ops))
    return {"outputs": read_outputs(launch, memory), "kernels": results}
