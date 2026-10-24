"""Graph IR (legalized for tiny-gpu) -> Kernel IR, one kernel per op (fused ops stay one kernel).

Mapping: one thread per output element. For a rank-2 result [M, N] the grid is M blocks of N
threads, so block = row, thread = column, and the global thread id is the flat row-major index.
Rank-1 results use one block. Addresses are word addresses from the memory plan, baked in as
immediates (tiny-gpu has no base+offset addressing, so each access computes its address).

This is the straightforward (-O0) code: loops are not unrolled and addresses are recomputed on
every iteration.
"""

from __future__ import annotations

from tinyc.gpu.kernel_ir import (
    Assign,
    For,
    If,
    Kernel,
    KernelBuilder,
    Load,
    Special,
    Store,
    verify_kernel,
)
from tinyc.gpu.memplan import MemPlan
from tinyc.ir.graph import Func, Op
from tinyc.ir.types import epilogue

BLOCK, THREAD, TID = Special("block"), Special("thread"), Special("tid")


def _grid(shape: tuple[int, ...]) -> tuple[int, int]:
    return (shape[0], shape[1]) if len(shape) == 2 else (1, shape[0])


def _index(operand_shape, result_shape):
    """Which special register indexes an elementwise operand: the flat id, or the column."""
    return TID if operand_shape == result_shape else THREAD


def _lower_matmul(op: Op, plan: MemPlan, name: str) -> Kernel:
    b = KernelBuilder()
    a_val, b_val = op.operands[0], op.operands[1]
    stages = epilogue(op.attrs)
    (_, k_dim), n_dim = a_val.type.shape, b_val.type.shape[1]
    a_base, b_base, c_base = plan.addr(a_val.name), plan.addr(b_val.name), plan.addr(op.result.name)
    loc = op.loc
    acc, k = b.fresh(), b.fresh()
    body: list = [Assign("set", acc, (0,), loc)]

    t_a, x, t_b, y, p = (b.fresh() for _ in range(5))
    loop = [
        Assign("mul", t_a, (BLOCK, k_dim), loc),  # &A[row][k] = a_base + row*K + k
        Assign("add", t_a, (t_a, k), loc),
        Assign("add", t_a, (t_a, a_base), loc),
        Load(x, t_a, loc),
        Assign("mul", t_b, (k, n_dim), loc),  # &B[k][col] = b_base + k*N + col
        Assign("add", t_b, (t_b, THREAD), loc),
        Assign("add", t_b, (t_b, b_base), loc),
        Load(y, t_b, loc),
        Assign("mul", p, (x, y), loc),
        Assign("add", acc, (acc, p), loc),
    ]
    body.append(For(k, 0, k_dim, loop, loc))
    if "bias" in stages:
        t_bias, z = b.fresh(), b.fresh()
        bias_base = plan.addr(op.operands[2].name)
        body += [
            Assign("add", t_bias, (THREAD, bias_base), loc),
            Load(z, t_bias, loc),
            Assign("add", acc, (acc, z), loc),
        ]
    if "relu" in stages:
        body.append(If(acc, 0, [Assign("set", acc, (0,), loc)], loc))
    t_c = b.fresh()
    body += [Assign("add", t_c, (TID, c_base), loc), Store(t_c, acc, loc)]
    return Kernel(name, _grid(op.result.type.shape), body, op.result.name, "matmul", loc)


def _lower_add(op: Op, plan: MemPlan, name: str) -> Kernel:
    b = KernelBuilder()
    shape = op.result.type.shape
    t_a, x, t_b, y, s, t_c = (b.fresh() for _ in range(6))
    lhs, rhs = op.operands
    loc = op.loc
    body = [
        Assign("add", t_a, (_index(lhs.type.shape, shape), plan.addr(lhs.name)), loc),
        Load(x, t_a, loc),
        Assign("add", t_b, (_index(rhs.type.shape, shape), plan.addr(rhs.name)), loc),
        Load(y, t_b, loc),
        Assign("add", s, (x, y), loc),
        Assign("add", t_c, (TID, plan.addr(op.result.name)), loc),
        Store(t_c, s, loc),
    ]
    return Kernel(name, _grid(shape), body, op.result.name, "add", loc)


def _lower_relu(op: Op, plan: MemPlan, name: str) -> Kernel:
    b = KernelBuilder()
    t_a, x, t_c = (b.fresh() for _ in range(3))
    loc = op.loc
    body = [
        Assign("add", t_a, (TID, plan.addr(op.operands[0].name)), loc),
        Load(x, t_a, loc),
        If(x, 0, [Assign("set", x, (0,), loc)], loc),  # relu: clamp x <= 0 to 0
        Assign("add", t_c, (TID, plan.addr(op.result.name)), loc),
        Store(t_c, x, loc),
    ]
    return Kernel(name, _grid(op.result.type.shape), body, op.result.name, "relu", loc)


_LOWER = {"matmul": _lower_matmul, "add": _lower_add, "relu": _lower_relu}


def lower(func: Func, plan: MemPlan) -> list[Kernel]:
    kernels = []
    for op in func.ops:
        if op.kind == "const":
            continue  # constants are the memory's initial contents, not kernels
        kernel = _LOWER[op.kind](op, plan, f"k{len(kernels)}_{op.kind}")
        verify_kernel(kernel)
        kernels.append(kernel)
    return kernels
