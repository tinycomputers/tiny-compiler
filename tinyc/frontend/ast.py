"""AST for `.tiny`. Every node carries its source span."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass

from tinyc.frontend.diagnostics import Span


@dataclass(frozen=True)
class TypeAnn:
    dtype: str
    shape: tuple[int, ...]
    span: Span


@dataclass(frozen=True)
class IntLit:
    value: int
    span: Span


@dataclass(frozen=True)
class ListLit:
    items: tuple[IntLit | ListLit, ...]
    span: Span


@dataclass(frozen=True)
class Name:
    name: str
    span: Span


@dataclass(frozen=True)
class MatMul:
    lhs: Expr
    rhs: Expr
    span: Span


@dataclass(frozen=True)
class Add:
    lhs: Expr
    rhs: Expr
    span: Span


@dataclass(frozen=True)
class Relu:
    arg: Expr
    span: Span


Expr = Name | MatMul | Add | Relu


@dataclass(frozen=True)
class InputDecl:
    name: str
    type: TypeAnn
    span: Span


@dataclass(frozen=True)
class ConstDecl:
    name: str
    type: TypeAnn
    value: ListLit | IntLit
    span: Span


@dataclass(frozen=True)
class Assign:
    name: str
    expr: Expr
    span: Span


@dataclass(frozen=True)
class Output:
    names: tuple[Name, ...]
    span: Span


Stmt = InputDecl | ConstDecl | Assign | Output


@dataclass(frozen=True)
class Program:
    stmts: tuple[Stmt, ...]


def dump(node, indent: int = 0) -> str:
    """Readable tree dump (spans omitted) for `tinyc --emit ast` and the web UI."""
    pad = "  " * indent
    if isinstance(node, tuple):
        return "\n".join(dump(n, indent) for n in node)
    if isinstance(node, IntLit):
        return f"{pad}{node.value}"
    if isinstance(node, TypeAnn):
        return f"{pad}{node.dtype}{list(node.shape)}"
    if not is_dataclass(node):
        return f"{pad}{node!r}"
    scalars = [
        f"{f.name}={getattr(node, f.name)!r}"
        for f in fields(node)
        if f.name != "span"
        and not is_dataclass(getattr(node, f.name))
        and not isinstance(getattr(node, f.name), tuple)
    ]
    lines = [f"{pad}{type(node).__name__}({', '.join(scalars)})"]
    for f in fields(node):
        v = getattr(node, f.name)
        if f.name != "span" and (is_dataclass(v) or isinstance(v, tuple)):
            lines.append(dump(v, indent + 1))
    return "\n".join(line for line in lines if line)
