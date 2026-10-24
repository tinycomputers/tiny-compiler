"""Reference interpreter for the Kernel IR, with the same dynamic counts the simulator reports.

It runs the *structured* kernels (before register allocation and assembly), thread by thread,
so it is independent of the lowering to tiny-gpu instructions and of the allocator. It also counts
what tiny-gpu would execute, so a test can require interpreter counts == simulator counts exactly:
that checks lowering, allocation and emission together.

Counting follows the idioms in `kernel_ir.flatten` and tiny-gpu's `step()`: a thread's cycle ends
at a taken branch, so a thread needs (taken branches + 1) cycles and a launch needs the maximum
over its threads.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from tinyc.gpu.kernel_ir import I32_HI, I32_LO, SPECIAL_REGS, Assign, For, Load, Special, Store
from tinyc.ir.errors import InputError, RangeError


class GpuFault(Exception):
    """A thread touched memory outside [0, mem_size)."""


@dataclass
class KernelCounts:
    name: str
    threads: int
    cycles: int  # max over threads of (taken branches + 1)
    ops: Counter = field(default_factory=Counter)  # tiny-gpu op name -> executions, all threads

    @property
    def instructions(self) -> int:
        return sum(self.ops.values())


def _run_thread(kernel, memory, env, ops: Counter) -> int:
    """Execute one thread; returns its taken-branch count."""
    taken = 0

    def val(x):
        if isinstance(x, Special):
            return env[x.name]
        return env[x] if not isinstance(x, int) else x

    def fit(v: int, what: str) -> int:
        if not I32_LO <= v <= I32_HI:
            raise RangeError(f"{what} overflows i32 in kernel @{kernel.name}")
        return v

    def check_addr(a: int) -> int:
        if not 0 <= a < len(memory):
            raise GpuFault(f"kernel @{kernel.name}: address {a} outside memory [0, {len(memory)})")
        return a

    def walk(body):
        nonlocal taken
        for s in body:
            if isinstance(s, Assign):
                ops[s.op.upper()] += 1
                a = [val(x) for x in s.args]
                env[s.dst] = fit(
                    a[0] if s.op == "set" else a[0] + a[1] if s.op == "add" else a[0] * a[1], s.op
                )
            elif isinstance(s, Load):
                ops["LD"] += 1
                env[s.dst] = memory[check_addr(val(s.addr))]
            elif isinstance(s, Store):
                ops["ST"] += 1
                memory[check_addr(val(s.addr))] = fit(val(s.src), "st")
            elif isinstance(s, For):
                ops["SET"] += 1
                env[s.var] = s.start
                for it in range(s.stop - s.start):
                    walk(s.body)
                    env[s.var] = env[s.var] + 1
                    ops["ADD"] += 1
                    ops["CMP"] += 1
                    ops["BRLT"] += 1
                    if it < s.stop - s.start - 1:
                        taken += 1  # the back edge is taken on every iteration but the last
            else:
                ops["CMP"] += 1
                ops["BRGT"] += 1
                if val(s.a) <= val(s.b):
                    walk(s.body)
                else:
                    taken += 1  # skip branch taken

    walk(kernel.body)
    return taken


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
    memory = initial_memory(prog.launch, inputs)
    results = []
    for k in prog.kernels:
        ops: Counter = Counter()
        worst = 0
        blocks, tpb = k.grid
        for block in range(blocks):
            for t in range(tpb):
                env = {"block": block, "thread": t, "tid": block * tpb + t}
                worst = max(worst, _run_thread(k, memory, env, ops))
        results.append(KernelCounts(k.name, k.threads, worst + 1, ops))
    return {"outputs": read_outputs(prog.launch, memory), "kernels": results}


assert set(SPECIAL_REGS) == {"block", "thread", "tid"}
