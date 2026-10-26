"""tiny-gpu backend: Graph IR -> memory plan -> Kernel IR -> registers -> `.tgpu` + launch JSON."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from tinyc.gpu import cost, interp
from tinyc.gpu.emit import emit_asm, to_launch_json
from tinyc.gpu.kernel_ir import Kernel, KernelError, flatten, print_kernel, verify_kernel
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

    @property
    def estimate(self) -> dict:
        regs = {k.name: self.registers.get(k.name, 0) for k in self.kernels}
        return cost.estimate(self.kernels, regs)

    @property
    def launch(self) -> dict:
        return to_launch_json(self)

    def asm_text(self) -> str:
        return "\n".join(self.asm[k.name] for k in self.kernels)


def _kernel_text(kernels: list[Kernel]) -> str:
    return "\n".join(print_kernel(k) for k in kernels)


def _problem(want, got_outputs, names) -> str | None:
    for name, w in zip(names, want):
        if got_outputs[name].shape != w.shape or not (got_outputs[name] == w).all():
            return f"output '{name}' differs from the oracle on seeded random inputs"
    return None


def compile_gpu_program(
    func: Func,
    ctx: PassContext | None = None,
    *,
    verify_each: bool = False,
    dump: Callable[[str, str], None] | None = None,
    seed: int = 42,
    num_checks: int = 3,
) -> GpuProgram:
    """Lower a gpu-legalized Graph IR function. Kernel code is the straightforward (-O0) form."""
    ctx = ctx or PassContext()
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
    for inputs, want in expectations:  # the structured kernels against the oracle
        got = interp.run_program(prog, inputs)["outputs"]
        if why := _problem(want, got, outputs):
            raise PassVerifyError("lower-to-kernel", why, "", text)

    for k in prog.kernels:
        try:
            verify_kernel(k)
            insts = flatten(k)
            alloc = allocate(insts, prog.num_registers, k.name)
        except RegisterPressure as e:
            raise PassError(e.message, e.loc) from None
        except KernelError as e:  # pragma: no cover - lowering never builds an invalid kernel
            raise PassVerifyError("lower-to-kernel", str(e), "", text) from None
        prog.asm[k.name] = emit_asm(k, insts, alloc)
        prog.registers[k.name] = alloc.registers_used
    asm = prog.asm_text()
    if dump:
        dump("regalloc-emit", asm)

    if verify_each:  # the emitted assembly on the real simulator against the oracle
        from runtime.gpu_runner import run_launch

        for inputs, want in expectations:
            try:
                res = run_launch(prog.launch, inputs)
            except (RangeError, OverflowError) as e:
                raise PassVerifyError(
                    "regalloc-emit", f"simulator failed: {e}", text, asm
                ) from None
            if not res.reserved_intact:
                raise PassVerifyError("regalloc-emit", "a kernel clobbered R5-R7", text, asm)
            if why := _problem(want, res.outputs, outputs):
                raise PassVerifyError("regalloc-emit", why, text, asm)
    return prog
