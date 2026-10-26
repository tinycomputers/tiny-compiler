"""Pipelines: which passes run, in what order, for a target and optimization level."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from tinyc.frontend import compile_to_graph
from tinyc.gpu import GpuProgram, compile_gpu_program
from tinyc.gpu.kernel_passes import KERNEL_PASSES
from tinyc.ir.graph import Func
from tinyc.passes import REGISTRY, Fixpoint, PassContext, PassManager, PassReport
from tinyc.tpu import compile_tile_ir
from tinyc.tpu.emit import to_json
from tinyc.tpu.tile_ir import Schedule

# These enable each other, so they run as one fixpoint group (see passes.manager.Fixpoint).
SIMPLIFY = ("canonicalize", "const-fold", "cse", "dce")


def pipeline_names(target: str | None = None, opt_level: int = 1) -> list[str]:
    """-O0 only type-checks and proves ranges; -O1 adds the simplifications and fusion.

    The simplify group (`SIMPLIFY`) runs to a fixpoint; names are listed once, in group order.
    """
    names = ["shape-infer"]
    if opt_level >= 1:
        names += [*SIMPLIFY, "fuse"]
    names.append("range-check")
    if target in ("tpu", "gpu"):
        names.append(f"legalize-{target}")
    return names


def build_pipeline(
    target: str | None = None, opt_level: int = 1, disabled: Iterable[str] = ()
) -> list:
    off = set(disabled)
    unknown = off - set(REGISTRY) - set(KERNEL_PASSES)  # kernel passes are disabled the same way
    if unknown:
        known = sorted(set(REGISTRY) | set(KERNEL_PASSES))
        raise ValueError(f"unknown passes: {sorted(unknown)} (known: {known})")
    out: list = []
    group: list = []
    for n in pipeline_names(target, opt_level):
        if n in off:
            continue
        if n in SIMPLIFY:
            group.append(REGISTRY[n]())
            continue
        if group:
            out.append(Fixpoint(group))
            group = []
        out.append(REGISTRY[n]())
    if group:  # (never reached: fuse/range-check always follow) kept for safety
        out.append(Fixpoint(group))
    return out


def compile_graph(
    text: str,
    filename: str = "<input>",
    *,
    target: str | None = None,
    opt_level: int = 1,
    disabled: Iterable[str] = (),
    verify_each: bool = False,
    ctx: PassContext | None = None,
    dump: Callable[[str, str], None] | None = None,
) -> tuple[Func, PassReport]:
    """Source text -> optimized, legalized Graph IR (the input to the target backends)."""
    func = compile_to_graph(text, filename)
    manager = PassManager(
        build_pipeline(target, opt_level, disabled), verify_each=verify_each, dump=dump
    )
    return func, manager.run(func, ctx)


@dataclass
class TpuCompilation:
    func: Func  # the optimized, tpu-legalized Graph IR
    schedule: Schedule  # Tile IR
    program: dict  # the `.ttpu.json` document
    report: PassReport  # graph pass records


def compile_tpu(
    text: str,
    filename: str = "<input>",
    *,
    opt_level: int = 1,
    disabled: Iterable[str] = (),
    verify_each: bool = False,
    ctx: PassContext | None = None,
    dump: Callable[[str, str], None] | None = None,
) -> TpuCompilation:
    """Source text -> `.ttpu.json` for tiny-tpu (graph passes, then lowering and tile passes)."""
    func, report = compile_graph(
        text, filename, target="tpu", opt_level=opt_level, disabled=disabled,
        verify_each=verify_each, ctx=ctx, dump=dump,
    )  # fmt: skip
    schedule = compile_tile_ir(func, opt_level, verify_each=verify_each, dump=dump)
    return TpuCompilation(func, schedule, to_json(schedule), report)


@dataclass
class GpuCompilation:
    func: Func  # the optimized, gpu-legalized Graph IR
    program: GpuProgram  # kernels, memory plan, assembly
    report: PassReport  # graph pass records

    @property
    def launch(self) -> dict:
        return self.program.launch


def compile_gpu(
    text: str,
    filename: str = "<input>",
    *,
    opt_level: int = 1,
    disabled: Iterable[str] = (),
    verify_each: bool = False,
    ctx: PassContext | None = None,
    dump: Callable[[str, str], None] | None = None,
) -> GpuCompilation:
    """Source text -> kernels and `.tgpu` assembly for tiny-gpu (graph passes, then the backend)."""
    ctx = ctx or PassContext()
    func, report = compile_graph(
        text, filename, target="gpu", opt_level=opt_level, disabled=disabled,
        verify_each=verify_each, ctx=ctx, dump=dump,
    )  # fmt: skip
    program = compile_gpu_program(
        func, ctx, opt_level=opt_level, disabled=disabled, verify_each=verify_each, dump=dump
    )
    return GpuCompilation(func, program, report)
