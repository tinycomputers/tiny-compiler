"""Graph IR: a single straight-line function in SSA form. Values are compared by identity."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from tinyc.ir.loc import Loc
from tinyc.ir.types import TensorType, infer_result_type


@dataclass(eq=False)
class Value:
    name: str  # printed as %name
    type: TensorType

    def __repr__(self) -> str:
        return f"%{self.name}: {self.type}"


@dataclass(eq=False)
class Op:
    kind: str  # "const" | "matmul" | "add" | "relu"
    operands: list[Value]
    result: Value
    loc: Loc | None = None
    data: np.ndarray | None = None  # const only (int64)


@dataclass(eq=False)
class Func:
    name: str
    args: list[Value] = field(default_factory=list)
    ops: list[Op] = field(default_factory=list)
    outputs: list[Value] = field(default_factory=list)


class FuncBuilder:
    """Builds a Func, inferring result types and naming results %0, %1, ..."""

    def __init__(self, name: str):
        self.func = Func(name)
        self._next = 0

    def add_arg(self, name: str, type: TensorType) -> Value:
        v = Value(name, type)
        self.func.args.append(v)
        return v

    def _emit(self, kind, operands, type, loc, data=None) -> Value:
        result = Value(str(self._next), type)
        self._next += 1
        self.func.ops.append(Op(kind, list(operands), result, loc, data))
        return result

    def op(self, kind: str, operands: Sequence[Value], loc: Loc | None = None) -> Value:
        """Append matmul/add/relu. Raises OpTypeError when the operands do not type-check."""
        return self._emit(kind, operands, infer_result_type(kind, [v.type for v in operands]), loc)

    def const(self, data: np.ndarray, type: TensorType, loc: Loc | None = None) -> Value:
        return self._emit("const", [], type, loc, np.asarray(data, dtype=np.int64))

    def finish(self, outputs: Sequence[Value]) -> Func:
        self.func.outputs = list(outputs)
        return self.func
