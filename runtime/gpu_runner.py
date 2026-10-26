"""Run `.tgpu` assembly on the unmodified tiny-gpu simulator.

We always go through tiny-gpu's own `assemble_file` and `TinyGPU`; the only thing added is a
counting wrapper around the `INSTRUCTIONS` table, which observes calls and changes no semantics.
"""

from __future__ import annotations

import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
from tinygpu import instructions as _instructions
from tinygpu.assembler import assemble_file
from tinygpu.gpu import TinyGPU

# R5 = block_id, R6 = thread_in_block, R7 = global tid are set by the simulator (PLAN §2.1).
RESERVED_REGS = (5, 6, 7)


@dataclass
class GpuResult:
    memory: np.ndarray
    cycles: int  # tiny-gpu "cycles": a cycle ends at a branch/jump/sync, not at every instruction
    instructions: int  # thread-instructions executed; use this next to `cycles` (PLAN §2.1)
    op_counts: Counter = field(default_factory=Counter)
    reserved_intact: bool = True  # R5 (block), R6 (thread), R7 (tid) still hold what the sim set

    def read(self, addr: int, length: int) -> list[int]:
        return [int(v) for v in self.memory[addr : addr + length]]


@contextmanager
def _counting(counts: Counter):
    table = _instructions.INSTRUCTIONS  # gpu.py looks ops up in this same dict object
    original = dict(table)

    def wrap(name, fn):
        def counted(*args, **kwargs):
            counts[name] += 1
            return fn(*args, **kwargs)

        return counted

    try:
        for name, fn in original.items():
            table[name] = wrap(name, fn)
        yield
    finally:
        table.clear()
        table.update(original)


def run_tgpu(
    source: str,
    *,
    grid: tuple[int, int],
    mem_size: int,
    memory_init: Mapping[int, Sequence[int]] | None = None,
    num_registers: int = 16,
    shared_size: int = 0,
    args: Sequence[int] | None = None,
    max_cycles: int = 100_000,
) -> GpuResult:
    """Assemble `source` and run it as one kernel launch over `grid=(num_blocks, threads_per_block)`.

    `memory_init` maps a word address to the values stored there. The simulator is built at its
    final thread count up front (PLAN §2.1: `set_grid` resizing leaves R7 = 0 for added threads).
    """
    num_blocks, tpb = grid
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "kernel.tgpu")
        with open(path, "w") as f:
            f.write(source)
        program, labels = assemble_file(path)

    gpu = TinyGPU(num_threads=num_blocks * tpb, num_registers=num_registers, mem_size=mem_size)
    for addr, values in (memory_init or {}).items():
        gpu.memory[addr : addr + len(values)] = np.asarray(values, dtype=np.int32)

    gpu.load_kernel(
        program, labels=labels, grid=(num_blocks, tpb), args=args, shared_size=shared_size
    )

    counts: Counter = Counter()
    with _counting(counts):
        gpu.run_kernel(max_cycles=max_cycles)

    if gpu.active.any():
        raise RuntimeError(f"kernel still running after {max_cycles} cycles")
    tid = np.arange(num_blocks * tpb)
    expected = {5: tid // tpb, 6: tid % tpb, 7: tid}
    intact = all((gpu.registers[:, r] == want).all() for r, want in expected.items())
    return GpuResult(
        memory=gpu.memory.copy(),
        cycles=len(gpu.history_registers),
        instructions=sum(counts.values()),
        op_counts=counts,
        reserved_intact=intact,
    )


@dataclass
class LaunchResult:
    outputs: dict[str, np.ndarray]  # int64, original shapes
    kernels: list[tuple[str, GpuResult]]  # (kernel name, its measurements), in launch order
    reserved_intact: bool

    @property
    def cycles(self) -> int:
        return sum(r.cycles for _, r in self.kernels)

    @property
    def instructions(self) -> int:
        return sum(r.instructions for _, r in self.kernels)

    @property
    def op_counts(self) -> Counter:
        total: Counter = Counter()
        for _, r in self.kernels:
            total.update(r.op_counts)
        return total


def run_launch(launch: Mapping, inputs: Mapping[str, object]) -> LaunchResult:
    """Run a launch document (`docs/TGPU_FORMAT.md`) on the real tiny-gpu simulator.

    Global memory persists across kernels; each kernel gets a fresh simulator built at its final
    thread count (tiny-gpu's `set_grid` resize leaves R7 = 0 for threads it adds).
    """
    memory = np.zeros(launch["mem_size"], dtype=np.int64)
    for name, b in launch["buffers"].items():
        if b["kind"] == "const":
            flat = np.asarray(b["data"], dtype=np.int64).reshape(-1)
        elif b["kind"] == "input":
            arr = np.asarray(inputs[name], dtype=np.int64)
            if list(arr.shape) != b["shape"]:
                raise ValueError(f"input '{name}' must have shape {b['shape']}")
            flat = arr.reshape(-1)
        else:
            continue
        memory[b["addr"] : b["addr"] + len(flat)] = flat

    results, intact = [], True
    for k in launch["kernels"]:
        res = run_tgpu(
            k["asm"],
            grid=tuple(k["grid"]),
            mem_size=launch["mem_size"],
            memory_init={0: memory},
            num_registers=launch["num_registers"],
            shared_size=k["shared_size"],
        )
        memory = res.memory.astype(np.int64)
        intact = intact and res.reserved_intact
        results.append((k["name"], res))

    outputs = {}
    for name in launch["outputs"]:
        b = launch["buffers"][name]
        size = int(np.prod(b["shape"]))
        outputs[name] = memory[b["addr"] : b["addr"] + size].reshape(b["shape"])
    return LaunchResult(outputs, results, intact)
