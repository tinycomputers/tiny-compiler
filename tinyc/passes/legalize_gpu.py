"""tiny-gpu legality: launch size and memory budget (one thread per output element).

The memory check is a conservative upper bound (every value gets its own buffer); the memory
planner in the GPU backend reuses buffers and will never need more.
"""

from tinyc.ir.errors import PassError


class LegalizeGpu:
    name = "legalize-gpu"

    def run(self, func, ctx):
        if ctx.gpu_num_registers < 8:
            raise PassError(
                f"tiny-gpu needs at least 8 registers (R7 holds the thread id); got {ctx.gpu_num_registers}"
            )
        for op in func.ops:
            if op.kind != "const" and op.result.type.size > ctx.gpu_max_threads:
                raise PassError(
                    f"{op.kind} result {op.result.type} needs {op.result.type.size} threads, "
                    f"more than tiny-gpu launches ({ctx.gpu_max_threads})",
                    op.loc,
                )
        words = sum(v.type.size for v in func.args) + sum(o.result.type.size for o in func.ops)
        if words > ctx.gpu_mem_words:
            raise PassError(
                f"program needs up to {words} words of tiny-gpu memory, budget is "
                f"{ctx.gpu_mem_words}"
            )
