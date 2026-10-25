"""tiny-gpu backend: Graph IR -> memory plan -> Kernel IR -> kernel passes -> registers -> `.tgpu`."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np

from tinyc.gpu import cost, interp
from tinyc.gpu.emit import emit_asm, to_launch_json
from tinyc.gpu.kernel_ir import Kernel, KernelError, flatten, print_kernel, verify_kernel
from tinyc.gpu.kernel_passes import KERNEL_PASSES, kernel_pass_names
from tinyc.gpu.lower import lower
from tinyc.gpu.memplan import MemPlan, check_plan, plan_memory
from tinyc.gpu.regalloc import RegisterPressure, allocate
from tinyc.ir import ref
from tinyc.ir.errors import PassError, RangeError
from tinyc.ir.graph import Func
from tinyc.passes.manager import PassContext, PassVerifyError


@dataclass
class GpuProgram:
    func: Func
    plan: MemPlan
    kernels: list[Kernel]
    num_registers: int
    consts: dict[str, np.ndarray]
    asm: dict[str, str] = field(default_factory=dict)  # kernel name -> .tgpu text
    registers: dict[str, int] = field(default_factory=dict)  # kernel name -> registers used
    _estimate: dict | None = field(default=None, repr=False)

    @property
    def estimate(self) -> dict:
        """Static cost bounds. Cached once the program is fully compiled (registers known)."""
        if self._estimate is not None:
            return self._estimate
        return cost.estimate(
            self.kernels, {k.name: self.registers.get(k.name, 0) for k in self.kernels}
        )

    @property
    def launch(self) -> dict:
        return to_launch_json(self)

    @property
    def launch_core(self) -> dict:
        """The launch document without the (costly to compute) estimate: what runners need."""
        return to_launch_json(self, with_estimate=False)

    def asm_text(self) -> str:
        return "\n".join(self.asm[k.name] for k in self.kernels)


def _kernel_text(kernels: list[Kernel]) -> str:
    return "\n".join(print_kernel(k) for k in kernels)


def _problem(want, got_outputs, names) -> str | None:
    for name, w in zip(names, want):
        if got_outputs[name].shape != w.shape or not (got_outputs[name] == w).all():
            return f"output '{name}' differs from the oracle on seeded random inputs"
    return None


def _kernel_problem(prog: GpuProgram, expectations, outputs) -> str | None:
    """Interpret the structured kernels on seeded inputs and compare with the oracle."""
    for inputs, want in expectations:
        try:
            got = interp.run_program(prog, inputs)["outputs"]
        except (RangeError, interp.GpuFault) as e:
            return f"the kernels fail where the source program does not: {e}"
        if why := _problem(want, got, outputs):
            return why
    return None


def compile_gpu_program(
    func: Func,
    ctx: PassContext | None = None,
    *,
    opt_level: int = 0,
    disabled: Iterable[str] = (),
    verify_each: bool = False,
    dump: Callable[[str, str], None] | None = None,
    seed: int = 42,
    num_checks: int = 3,
) -> GpuProgram:
    """Lower a gpu-legalized Graph IR function, optimize its kernels, allocate registers, emit.

    Kernel optimization levels (see `kernel_passes`): -O0 none, -O1 licm + strength-reduce,
    -O2 adds shared-memory tiling, full unrolling and constant folding.
    """
    ctx = ctx or PassContext()
    off = set(disabled)
    expectations = ref.seeded_expectations(func, num_checks, seed) if verify_each else []
    plan = plan_memory(func)
    check_plan(func, plan)
    if plan.mem_size > ctx.gpu_mem_words:
        raise PassError(
            f"program needs {plan.mem_size} words of tiny-gpu memory, budget is {ctx.gpu_mem_words}"
        )
    consts = {op.result.name: op.data for op in func.ops if op.kind == "const"}
    prog = GpuProgram(func, plan, lower(func, plan), ctx.gpu_num_registers, consts)
    outputs = [v.name for v in func.outputs]

    text = _kernel_text(prog.kernels)
    if dump:
        dump("lower-to-kernel", text)
    if why := _kernel_problem(prog, expectations, outputs):
        raise PassVerifyError("lower-to-kernel", why, "", text)

    for name in kernel_pass_names(opt_level):
        if name in off:
            continue
        before = text
        for k in prog.kernels:
            KERNEL_PASSES[name]().run(k)
        text = _kernel_text(prog.kernels)
        try:
            for k in prog.kernels:
                verify_kernel(k)
        except KernelError as e:
            raise PassVerifyError(name, str(e), before, text) from None
        if dump:
            dump(name, text)
        if why := _kernel_problem(prog, expectations, outputs):
            raise PassVerifyError(name, why, before, text)

    for k in prog.kernels:
        try:
            insts = flatten(k)
            alloc = allocate(insts, prog.num_registers, k.name)
        except RegisterPressure as e:
            raise PassError(e.message, e.loc) from None
        prog.asm[k.name] = emit_asm(k, insts, alloc)
        prog.registers[k.name] = alloc.registers_used
    prog._estimate = cost.estimate(prog.kernels, prog.registers)
    asm = prog.asm_text()
    if dump:
        dump("regalloc-emit", asm)

    if verify_each:  # the emitted assembly on the real simulator against the oracle
        from runtime.gpu_runner import run_launch

        for inputs, want in expectations:
            try:
                res = run_launch(prog.launch_core, inputs)
            except (RangeError, OverflowError) as e:
                raise PassVerifyError(
                    "regalloc-emit", f"simulator failed: {e}", text, asm
                ) from None
            if not res.reserved_intact:
                raise PassVerifyError("regalloc-emit", "a kernel clobbered R5-R7", text, asm)
            if why := _problem(want, res.outputs, outputs):
                raise PassVerifyError("regalloc-emit", why, text, asm)
    return prog
