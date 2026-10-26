"""Recursive-descent parser for `.tiny`. Precedence: `@` binds tighter than `+`; both left-assoc."""

from __future__ import annotations

import dataclasses

from tinyc.frontend import ast
from tinyc.frontend.diagnostics import CompileError, Diagnostic, SourceFile, Span
from tinyc.frontend.lexer import Token, lex
from tinyc.ir.types import MAX_RANK


class Parser:
    def __init__(self, source: SourceFile):
        self.source = source
        self.tokens = lex(source)
        self.pos = 0

    # -- helpers ---------------------------------------------------------------------------
    @property
    def tok(self) -> Token:
        return self.tokens[self.pos]

    def error(self, message: str, span: Span, note: str | None = None) -> CompileError:
        return CompileError(self.source, Diagnostic(message, span, note))

    @staticmethod
    def describe(tok: Token) -> str:
        return {"NEWLINE": "end of line", "EOF": "end of file"}.get(tok.kind, f"'{tok.text}'")

    def expect(self, kind: str, what: str | None = None) -> Token:
        if self.tok.kind != kind:
            want = what or f"'{kind}'"
            raise self.error(f"expected {want}, found {self.describe(self.tok)}", self.tok.span)
        tok = self.tok
        self.pos += 1
        return tok

    # -- grammar ---------------------------------------------------------------------------
    def program(self) -> ast.Program:
        stmts = []
        while self.tok.kind != "EOF":
            stmts.append(self.stmt())
        return ast.Program(tuple(stmts))

    def stmt(self) -> ast.Stmt:
        first = self.tok
        if first.kind == "input":
            self.pos += 1
            name = self.expect("NAME", "a name")
            self.expect(":")
            stmt = ast.InputDecl(
                name.text, self.type_ann(), Span.join(first.span, self.prev_span())
            )
        elif first.kind == "const":
            self.pos += 1
            name = self.expect("NAME", "a name")
            self.expect(":")
            ty = self.type_ann()
            self.expect("=")
            value = self.literal()
            stmt = ast.ConstDecl(name.text, ty, value, Span.join(first.span, value.span))
        elif first.kind == "output":
            self.pos += 1
            names = [self.name()]
            while self.tok.kind == ",":
                self.pos += 1
                names.append(self.name())
            stmt = ast.Output(tuple(names), Span.join(first.span, names[-1].span))
        elif first.kind == "NAME":
            self.pos += 1
            self.expect("=")
            expr = self.expr()
            stmt = ast.Assign(first.text, expr, Span.join(first.span, expr.span))
        else:
            raise self.error(
                f"expected a statement ('input', 'const', 'output' or 'NAME = ...'), "
                f"found {self.describe(first)}",
                first.span,
            )
        if self.tok.kind != "NEWLINE":
            raise self.error(
                f"expected end of line, found {self.describe(self.tok)}", self.tok.span
            )
        self.pos += 1
        return stmt

    def prev_span(self) -> Span:
        return self.tokens[self.pos - 1].span

    def name(self) -> ast.Name:
        tok = self.expect("NAME", "a name")
        return ast.Name(tok.text, tok.span)

    def type_ann(self) -> ast.TypeAnn:
        if self.tok.kind not in ("i8", "i32"):
            raise self.error(
                f"expected a type ('i8' or 'i32'), found {self.describe(self.tok)}", self.tok.span
            )
        start = self.tok
        self.pos += 1
        self.expect("[")
        dims = [self.dim()]
        while self.tok.kind == ",":
            self.pos += 1
            dims.append(self.dim())
        close = self.expect("]")
        span = Span.join(start.span, close.span)
        if len(dims) > MAX_RANK:
            raise self.error(
                f"only rank 1 and 2 tensors are supported (got rank {len(dims)})", span
            )
        return ast.TypeAnn(start.kind, tuple(dims), span)

    def dim(self) -> int:
        tok = self.expect("INT", "a dimension")
        if int(tok.text) < 1:
            raise self.error("dimensions must be positive", tok.span)
        return int(tok.text)

    def literal(self) -> ast.ListLit | ast.IntLit:
        if self.tok.kind == "[":
            open_ = self.tok
            self.pos += 1
            items = [self.literal()]
            while self.tok.kind == ",":
                self.pos += 1
                items.append(self.literal())
            close = self.expect("]")
            return ast.ListLit(tuple(items), Span.join(open_.span, close.span))
        sign = self.tok if self.tok.kind == "-" else None
        if sign:
            self.pos += 1
        tok = self.expect("INT", "a number or '['")
        value = int(tok.text)
        return ast.IntLit(-value if sign else value, Span.join((sign or tok).span, tok.span))

    def expr(self) -> ast.Expr:
        lhs = self.term()
        while self.tok.kind == "+":
            self.pos += 1
            rhs = self.term()
            lhs = ast.Add(lhs, rhs, Span.join(lhs.span, rhs.span))
        return lhs

    def term(self) -> ast.Expr:
        lhs = self.atom()
        while self.tok.kind == "@":
            self.pos += 1
            rhs = self.atom()
            lhs = ast.MatMul(lhs, rhs, Span.join(lhs.span, rhs.span))
        return lhs

    def atom(self) -> ast.Expr:
        tok = self.tok
        if tok.kind == "NAME":
            self.pos += 1
            return ast.Name(tok.text, tok.span)
        if tok.kind == "relu":
            self.pos += 1
            self.expect("(")
            arg = self.expr()
            close = self.expect(")")
            return ast.Relu(arg, Span.join(tok.span, close.span))
        if tok.kind == "(":
            self.pos += 1
            inner = self.expr()
            close = self.expect(")")
            return dataclasses.replace(inner, span=Span.join(tok.span, close.span))
        raise self.error(f"expected an expression, found {self.describe(tok)}", tok.span)


def parse(source: SourceFile) -> ast.Program:
    return Parser(source).program()
