"""Pipelines: which passes run, in what order, for a target and optimization level."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from tinyc.frontend import compile_to_graph
from tinyc.ir.graph import Func
from tinyc.passes import REGISTRY, PassContext, PassManager, PassReport

OPTIMIZATIONS = ("canonicalize", "const-fold", "cse", "dce", "fuse")


def pipeline_names(target: str | None = None, opt_level: int = 1) -> list[str]:
    """-O0 only type-checks and proves ranges; -O1 adds the optimizations."""
    names = ["shape-infer"]
    if opt_level >= 1:
        names += ["canonicalize", "const-fold", "canonicalize", "cse", "dce", "fuse"]
    names.append("range-check")
    if target in ("tpu", "gpu"):
        names.append(f"legalize-{target}")
    return names


def build_pipeline(
    target: str | None = None, opt_level: int = 1, disabled: Iterable[str] = ()
) -> list:
    off = set(disabled)
    unknown = off - set(REGISTRY)
    if unknown:
        raise ValueError(f"unknown passes: {sorted(unknown)} (known: {sorted(REGISTRY)})")
    return [REGISTRY[n]() for n in pipeline_names(target, opt_level) if n not in off]


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
