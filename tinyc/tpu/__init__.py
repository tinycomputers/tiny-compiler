"""tiny-tpu backend: Graph IR -> Tile IR -> `.ttpu.json`."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from tinyc.ir import ref
from tinyc.ir.errors import RangeError
from tinyc.ir.graph import Func
from tinyc.passes.manager import PassVerifyError
from tinyc.tpu import interp
from tinyc.tpu.emit import to_json
from tinyc.tpu.lower import lower
from tinyc.tpu.tile_ir import Schedule, ScheduleError, print_schedule, verify_schedule
from tinyc.tpu.tile_passes import TILE_PASSES


def _expectations(func: Func, n: int, seed: int):
    out = []
    for i in range(n):
        inputs = ref.random_inputs(func, np.random.default_rng(seed + i))
        try:
            out.append((inputs, ref.evaluate(func, inputs)))
        except RangeError:
            continue
    return out


def _problem(s: Schedule, expectations) -> str | None:
    program = to_json(s)
    for inputs, want in expectations:
        try:
            got = interp.run_program(program, inputs)["outputs"]
        except RangeError as e:
            return f"now fails on inputs that used to work: {e}"
        for name, w in zip(s.outputs, want):
            if got[name].shape != w.shape or not (got[name] == w).all():
                return f"output '{name}' differs from the oracle on seeded random inputs"
    return None


def compile_tile_ir(
    func: Func,
    opt_level: int = 1,
    *,
    verify_each: bool = False,
    dump: Callable[[str, str], None] | None = None,
    seed: int = 42,
    num_checks: int = 4,
) -> Schedule:
    """Lower a tpu-legalized Graph IR function and run the tile passes."""
    expectations = _expectations(func, num_checks, seed) if verify_each else []
    schedule = lower(func)
    stages = [("lower-to-tile", lambda: None)]
    if opt_level >= 1:
        stages += [(n, (lambda p=p: p().run(schedule))) for n, p in TILE_PASSES.items()]
    before = ""
    for name, step in stages:
        step()
        text = print_schedule(schedule)
        try:
            verify_schedule(schedule)
        except ScheduleError as e:
            raise PassVerifyError(name, str(e), before, text) from None
        if verify_each and (why := _problem(schedule, expectations)):
            raise PassVerifyError(name, why, before, text)
        if dump:
            dump(name, text)
        before = text
    return schedule
