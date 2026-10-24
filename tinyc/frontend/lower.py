"""AST -> Graph IR, with name resolution and type checking.

Types are inferred while lowering through the same rules the verifier uses (`ir.types`).
Errors stop at the first problem and carry a span.
"""

from __future__ import annotations

import difflib

import numpy as np

from tinyc.frontend import ast
from tinyc.frontend.diagnostics import CompileError, Diagnostic, SourceFile, Span
from tinyc.ir.errors import OpTypeError
from tinyc.ir.graph import Func, FuncBuilder, Value
from tinyc.ir.loc import Loc
from tinyc.ir.types import RANGES, TensorType


class _Lowerer:
    def __init__(self, source: SourceFile):
        self.source = source
        self.b = FuncBuilder("main")
        self.env: dict[str, Value] = {}
        self.defined_at: dict[str, Span] = {}

    def error(self, message: str, span: Span, note: str | None = None) -> CompileError:
        return CompileError(self.source, Diagnostic(message, span, note))

    def loc(self, span: Span) -> Loc:
        return Loc(self.source.name, span.line, span.col)

    def define(self, name: str, value: Value, span: Span) -> None:
        if name in self.env:
            first = self.defined_at[name]
            raise self.error(
                f"'{name}' is already defined", span, f"first defined on line {first.line}"
            )
        self.env[name], self.defined_at[name] = value, span

    def lookup(self, node: ast.Name) -> Value:
        if node.name not in self.env:
            close = difflib.get_close_matches(node.name, self.env, n=1)
            raise self.error(
                f"undefined name '{node.name}'",
                node.span,
                f"did you mean '{close[0]}'?" if close else None,
            )
        return self.env[node.name]

    def expr(self, e: ast.Expr) -> Value:
        if isinstance(e, ast.Name):
            return self.lookup(e)
        kind, args = {
            ast.MatMul: ("matmul", lambda: (e.lhs, e.rhs)),
            ast.Add: ("add", lambda: (e.lhs, e.rhs)),
            ast.Relu: ("relu", lambda: (e.arg,)),
        }[type(e)]
        operands = [self.expr(x) for x in args()]
        try:
            return self.b.op(kind, operands, self.loc(e.span))
        except OpTypeError as err:
            raise self.error(str(err), e.span) from None

    def tensor_type(self, ann: ast.TypeAnn) -> TensorType:
        return TensorType(ann.dtype, ann.shape)

    def literal(self, lit: ast.ListLit | ast.IntLit, ty: TensorType) -> np.ndarray:
        lo, hi = RANGES[ty.dtype]
        out = np.zeros(ty.shape, dtype=np.int64)

        def fill(node, index: tuple[int, ...]) -> None:
            depth = len(index)
            if depth == ty.rank:
                if not isinstance(node, ast.IntLit):
                    raise self.error("expected a number here, found a list", node.span)
                if not lo <= node.value <= hi:
                    raise self.error(
                        f"value {node.value} does not fit in {ty.dtype}",
                        node.span,
                        f"{ty.dtype} holds {lo}..{hi}",
                    )
                out[index] = node.value
                return
            if not isinstance(node, ast.ListLit):
                raise self.error(
                    f"expected a list of {ty.shape[depth]} values, found a number", node.span
                )
            if len(node.items) != ty.shape[depth]:
                raise self.error(
                    f"expected a list of {ty.shape[depth]} values, found {len(node.items)}",
                    node.span,
                    f"declared type is {ty}",
                )
            for i, item in enumerate(node.items):
                fill(item, (*index, i))

        fill(lit, ())
        return out

    def stmt(self, s: ast.Stmt, outputs: list[Value]) -> None:
        if isinstance(s, ast.InputDecl):
            self.define(s.name, self.b.add_arg(s.name, self.tensor_type(s.type)), s.span)
        elif isinstance(s, ast.ConstDecl):
            ty = self.tensor_type(s.type)
            value = self.b.const(self.literal(s.value, ty), ty, self.loc(s.span))
            self.define(s.name, value, s.span)
        elif isinstance(s, ast.Assign):
            self.define(s.name, self.expr(s.expr), s.span)
        else:
            for n in s.names:
                v = self.lookup(n)
                if any(v is o for o in outputs):
                    raise self.error(f"'{n.name}' is already an output", n.span)
                outputs.append(v)

    def run(self, prog: ast.Program) -> Func:
        outputs: list[Value] = []
        for s in prog.stmts:
            self.stmt(s, outputs)
        if not outputs:
            span = prog.stmts[-1].span if prog.stmts else Span(1, 1, 1, 1)
            raise self.error("program has no outputs", span, "add a line like `output C`")
        return self.b.finish(outputs)


def lower(prog: ast.Program, source: SourceFile) -> Func:
    return _Lowerer(source).run(prog)
