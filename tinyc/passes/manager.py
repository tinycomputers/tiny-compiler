"""Pass manager: runs passes in order, verifies after each, and can translation-validate them.

* After every pass the IR is structurally verified (`ir.verifier`), always.
* With `verify_each`, the IR is also *run* after every pass on seeded random inputs (the
  independent scalar interpreter) and compared with the numpy oracle's answers for the program
  as it was before any pass. A failure names the pass and shows the IR before and after.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from tinyc.ir import interp, ref
from tinyc.ir.errors import RangeError, VerifyError
from tinyc.ir.graph import Func
from tinyc.ir.printer import print_func
from tinyc.ir.verifier import verify


@dataclass
class PassContext:
    """Options and results shared by passes."""

    # i32 *inputs* are assumed to lie in this range; range_check relies on it (i8 inputs are exact).
    i32_input_range: tuple[int, int] = (-(2**20), 2**20)
    gpu_max_threads: int = 1024  # tiny-gpu simulator speed cap
    gpu_mem_words: int = 65536  # global memory budget for the memory planner
    gpu_num_registers: int = 16  # tiny-gpu registers per thread (R5-R7 reserved by the simulator)
    analysis: dict = field(default_factory=dict)  # e.g. analysis["ranges"] from range_check


class Pass(Protocol):
    name: str

    def run(self, func: Func, ctx: PassContext) -> None: ...


class Fixpoint:
    """Run a group of passes repeatedly until a whole round changes nothing.

    Simplification passes enable each other (CSE can make operands non-canonical, folding can
    expose dead code, ...), so running each once leaves work behind and the pipeline would not be
    idempotent. Every sub-pass is still verified, recorded and dumped on every round.
    """

    name = "fixpoint"

    def __init__(self, passes: Sequence[Pass], max_iterations: int = 8):
        self.passes, self.max_iterations = list(passes), max_iterations


class PassVerifyError(Exception):
    def __init__(self, pass_name: str, reason: str, before: str, after: str):
        self.pass_name, self.reason, self.before, self.after = pass_name, reason, before, after
        super().__init__(
            f"pass '{pass_name}' broke the program: {reason}\n"
            f"--- IR before ---\n{before}--- IR after ---\n{after}"
        )


@dataclass
class PassRecord:
    name: str
    seconds: float
    changed: bool
    ir: str  # IR text after this pass


@dataclass
class PassReport:
    initial_ir: str
    records: list[PassRecord] = field(default_factory=list)

    def timing_table(self) -> str:
        rows = [f"{r.name:<14} {r.seconds * 1000:8.2f} ms  {'changed' if r.changed else '-'}"
                for r in self.records]  # fmt: skip
        return "\n".join(rows)


class PassManager:
    def __init__(
        self,
        passes: Sequence[Pass],
        *,
        verify_each: bool = False,
        num_checks: int = 4,
        seed: int = 42,
        dump: Callable[[str, str], None] | None = None,
    ):
        self.passes, self.verify_each = list(passes), verify_each
        self.num_checks, self.seed, self.dump = num_checks, seed, dump

    def _expectations(self, func: Func):
        return ref.seeded_expectations(func, self.num_checks, self.seed)

    def _semantic_problem(self, func: Func, expectations) -> str | None:
        for inputs, want in expectations:
            try:
                got = interp.run(func, inputs)
            except RangeError as e:
                return f"now fails on inputs that used to work: {e}"
            for i, (g, w) in enumerate(zip(got, want)):
                if g.shape != w.shape or not (g == w).all():
                    return f"output {i} differs from the oracle on seeded random inputs"
        return None

    def run(self, func: Func, ctx: PassContext | None = None) -> PassReport:
        ctx = ctx or PassContext()
        text = print_func(func)
        report = PassReport(initial_ir=text)
        expectations = self._expectations(func) if self.verify_each else []
        if self.dump:
            self.dump("input", text)

        def step(p: Pass) -> bool:
            """Run one pass with verification, recording and dumping. True if the IR changed."""
            nonlocal text
            start = time.perf_counter()
            p.run(func, ctx)
            seconds = time.perf_counter() - start
            after = print_func(func)
            try:
                verify(func)
            except VerifyError as e:
                raise PassVerifyError(p.name, "; ".join(e.problems), text, after) from None
            if self.verify_each and (why := self._semantic_problem(func, expectations)):
                raise PassVerifyError(p.name, why, text, after)
            changed = after != text
            report.records.append(PassRecord(p.name, seconds, changed, after))
            if self.dump:
                self.dump(p.name, after)
            text = after
            return changed

        for item in self.passes:
            if isinstance(item, Fixpoint):
                for _ in range(item.max_iterations):
                    changed = [step(p) for p in item.passes]  # every sub-pass runs each round
                    if not any(changed):
                        break
                else:
                    raise RuntimeError(
                        f"fixpoint group did not converge in {item.max_iterations} rounds"
                    )
            else:
                step(item)
        return report
