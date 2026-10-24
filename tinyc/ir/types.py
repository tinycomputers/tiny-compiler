"""Tensor types and the single source of truth for op typing rules.

The frontend builder and the verifier both call `infer_result_type`, so a rule can never
differ between "what lowering produced" and "what the verifier accepts".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from tinyc.ir.errors import OpTypeError

DTYPES = ("i8", "i32")
RANGES = {"i8": (-128, 127), "i32": (-(2**31), 2**31 - 1)}
MAX_RANK = 2
OP_ARITY = {"const": 0, "relu": 1, "matmul": 2, "add": 2}


@dataclass(frozen=True)
class TensorType:
    dtype: str
    shape: tuple[int, ...]

    def __str__(self) -> str:
        return f"{self.dtype}[{','.join(map(str, self.shape))}]"

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n


def type_problem(t: TensorType) -> str | None:
    """Why `t` is not a legal v1 type, or None."""
    if t.dtype not in DTYPES:
        return f"unknown element type '{t.dtype}'"
    if not 1 <= t.rank <= MAX_RANK:
        return f"only rank 1 and 2 tensors are supported (got rank {t.rank})"
    if any(d < 1 for d in t.shape):
        return f"dimensions must be positive (got {t})"
    return None


def infer_result_type(kind: str, operands: Sequence[TensorType]) -> TensorType:
    """Result type of a non-const op, or OpTypeError with a user-facing message."""
    if kind == "relu":
        (x,) = operands
        return x
    a, b = operands
    if kind == "matmul":
        if a.rank != 2 or b.rank != 2:
            raise OpTypeError(f"matmul needs rank-2 operands (got {a} @ {b})")
        if a.shape[1] != b.shape[0]:
            raise OpTypeError(
                f"matmul inner dimensions differ: {a.shape[1]} vs {b.shape[0]} ({a} @ {b})"
            )
        return TensorType("i32", (a.shape[0], b.shape[1]))
    if kind == "add":
        if a.shape == b.shape:
            return TensorType("i32", a.shape)
        for m, v in ((a, b), (b, a)):  # [M,N] + [N] broadcasts the row
            if m.rank == 2 and v.rank == 1 and m.shape[1] == v.shape[0]:
                return TensorType("i32", m.shape)
        raise OpTypeError(
            f"add shapes are incompatible: {a} + {b} (shapes must match, or be [M,N] + [N])"
        )
    raise OpTypeError(f"unknown op '{kind}'")
