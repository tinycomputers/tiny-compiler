"""Kernel -> `.tgpu` assembly text, and the whole program -> launch JSON (docs/TGPU_FORMAT.md)."""

from __future__ import annotations

import json

from tinyc.gpu.kernel_ir import Kernel, VInst
from tinyc.gpu.regalloc import Allocation, reg_name

FORMAT = "tgpu-launch"
VERSION = 1


def _inst_text(i: VInst, asg) -> str:
    r = lambda x: reg_name(x, asg)
    if i.op == "LABEL":
        return f"{i.label}:"
    if i.op in ("BRLT", "BRGT"):
        return f"{i.op} {i.label}"
    if i.op == "SYNCB":
        return "SYNCB"
    if i.op in ("ST", "SHST"):
        return f"{i.op} {r(i.srcs[0])}, {r(i.srcs[1])}"
    if i.op == "CMP":
        return f"CMP {r(i.srcs[0])}, {r(i.srcs[1])}"
    return f"{i.op} {r(i.dst)}, " + ", ".join(r(s) for s in i.srcs)


def emit_asm(kernel: Kernel, insts: list[VInst], alloc: Allocation) -> str:
    """tiny-gpu assembly. Comments (`;`) carry the source line:col each instruction came from."""
    where = f"{kernel.loc.file}:{kernel.loc.line}:{kernel.loc.col}" if kernel.loc else "?"
    shared = f"; {kernel.shared_size} shared words per block" if kernel.shared_size else ""
    lines = [
        f"; kernel {kernel.name}: %{kernel.value} = {kernel.kind}  ({where})",
        (
            f"; grid: {kernel.grid[0]} blocks x {kernel.grid[1]} threads{shared};"
            f" {alloc.registers_used} registers (R5-R7 reserved, read-only)"
        ),
    ]
    texts = [_inst_text(i, alloc.assignment) for i in insts]
    width = max((len(t) for t, i in zip(texts, insts) if i.op != "LABEL"), default=0) + 2
    for text, i in zip(texts, insts):
        if i.op == "LABEL" or i.loc is None:
            lines.append(text)
        else:
            lines.append(f"{text.ljust(width)}; {i.loc.line}:{i.loc.col}")
    return "\n".join(lines) + "\n"


def to_launch_json(prog, with_estimate: bool = True) -> dict:
    """`prog` is a tinyc.gpu.GpuProgram."""
    buffers = {}
    for name, r in prog.plan.buffers.items():
        entry = {"addr": r.addr, "dtype": r.dtype, "shape": list(r.shape), "kind": r.kind}
        if r.kind == "const":
            entry["data"] = prog.consts[name].tolist()
        buffers[name] = entry
    return {
        "format": FORMAT,
        "version": VERSION,
        "name": prog.func.name,
        "num_registers": prog.num_registers,
        "mem_size": prog.plan.mem_size,
        "buffers": buffers,
        "inputs": [a.name for a in prog.func.args],
        "outputs": [v.name for v in prog.func.outputs],
        "kernels": [
            {
                "name": k.name,
                "value": k.value,
                "op": k.kind,
                "grid": list(k.grid),
                "shared_size": k.shared_size,
                "asm": prog.asm.get(k.name, ""),  # empty before assembly (validation of the IR)
            }
            for k in prog.kernels
        ],
        "estimate": prog.estimate if with_estimate else {},
    }


def dumps(launch: dict) -> str:
    return json.dumps(launch, indent=2) + "\n"
