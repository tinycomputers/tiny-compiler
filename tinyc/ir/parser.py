"""Parser for the textual Graph IR printed by `ir.printer` (round-trips exactly)."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from tinyc.frontend.diagnostics import CompileError, Diagnostic, SourceFile, Span
from tinyc.ir.graph import Func, Op, Value
from tinyc.ir.loc import Loc
from tinyc.ir.types import OP_ARITY, TensorType
from tinyc.ir.verifier import verify

_TOKEN = re.compile(
    r"(?P<ws>[ \t\r\n]+)|(?P<comment>//[^\n]*)|(?P<arrow>->)"
    r'|(?P<val>%[A-Za-z0-9_]+)|(?P<sym>@[A-Za-z_][A-Za-z0-9_]*)|(?P<str>"[^"\n]*")'
    r"|(?P<word>[A-Za-z_][A-Za-z0-9_]*)|(?P<int>-?[0-9]+)|(?P<punct>[:,=(){}\[\]])"
)


@dataclass(frozen=True)
class _Tok:
    kind: str
    text: str
    span: Span


def _lex(source: SourceFile) -> list[_Tok]:
    text, toks = source.text, []
    pos, line, line_start = 0, 1, 0
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        col = pos - line_start + 1
        if m is None:
            span = Span(line, col, line, col + 1)
            raise CompileError(source, Diagnostic(f"unexpected character {text[pos]!r}", span))
        kind, lexeme = m.lastgroup, m.group()
        if kind not in ("ws", "comment"):
            toks.append(_Tok(kind, lexeme, Span(line, col, line, col + len(lexeme))))
        for i, ch in enumerate(lexeme):
            if ch == "\n":
                line, line_start = line + 1, pos + i + 1
        pos = m.end()
    col = pos - line_start + 1
    toks.append(_Tok("eof", "", Span(line, col, line, col)))
    return toks


class _Parser:
    def __init__(self, source: SourceFile):
        self.source, self.toks, self.pos = source, _lex(source), 0
        self.values: dict[str, Value] = {}

    @property
    def tok(self) -> _Tok:
        return self.toks[self.pos]

    def error(self, message: str, span: Span) -> CompileError:
        return CompileError(self.source, Diagnostic(message, span))

    def take(self, kind: str, text: str | None = None) -> _Tok:
        t = self.tok
        if t.kind != kind or (text is not None and t.text != text):
            want = text or kind
            raise self.error(f"expected '{want}', found '{t.text or 'end of file'}'", t.span)
        self.pos += 1
        return t

    def at(self, kind: str, text: str | None = None) -> bool:
        return self.tok.kind == kind and (text is None or self.tok.text == text)

    def type(self) -> TensorType:
        dtype = self.take("word").text
        self.take("punct", "[")
        dims = [int(self.take("int").text)]
        while self.at("punct", ","):
            self.pos += 1
            dims.append(int(self.take("int").text))
        self.take("punct", "]")
        return TensorType(dtype, tuple(dims))

    def value_ref(self) -> Value:
        t = self.take("val")
        if t.text[1:] not in self.values:
            raise self.error(f"undefined value {t.text}", t.span)
        return self.values[t.text[1:]]

    def define(self, tok: _Tok, ty: TensorType) -> Value:
        name = tok.text[1:]
        if name in self.values:
            raise self.error(f"{tok.text} is already defined", tok.span)
        self.values[name] = Value(name, ty)
        return self.values[name]

    def literal(self):
        if self.at("punct", "["):
            self.pos += 1
            items = [self.literal()]
            while self.at("punct", ","):
                self.pos += 1
                items.append(self.literal())
            self.take("punct", "]")
            return items
        return int(self.take("int").text)

    def loc(self) -> Loc | None:
        if not self.at("word", "loc"):
            return None
        self.pos += 1
        self.take("punct", "(")
        file = self.take("str").text[1:-1]
        self.take("punct", ":")
        line = int(self.take("int").text)
        self.take("punct", ":")
        col = int(self.take("int").text)
        self.take("punct", ")")
        return Loc(file, line, col)

    def op(self) -> Op:
        res_tok = self.take("val")
        self.take("punct", "=")
        kind_tok = self.take("word")
        kind = kind_tok.text
        if kind not in OP_ARITY:
            raise self.error(f"unknown op '{kind}'", kind_tok.span)
        operands, data, lit_span = [], None, self.tok.span
        if kind == "const":
            lit = self.literal()
        else:
            operands = [self.value_ref()]
            while self.at("punct", ","):
                self.pos += 1
                operands.append(self.value_ref())
        self.take("punct", ":")
        ty = self.type()
        if kind == "const":
            try:
                data = np.array(lit, dtype=np.int64)
            except ValueError:
                raise self.error("const data is ragged", lit_span) from None
            if data.shape != ty.shape:
                raise self.error(
                    f"const data has shape {list(data.shape)}, type says {ty}", lit_span
                )
        return Op(kind, operands, self.define(res_tok, ty), self.loc(), data)

    def func(self) -> Func:
        self.take("word", "func")
        name = self.take("sym").text[1:]
        self.take("punct", "(")
        args: list[Value] = []
        while not self.at("punct", ")"):
            if args:
                self.take("punct", ",")
            tok = self.take("val")
            self.take("punct", ":")
            args.append(self.define(tok, self.type()))
        self.take("punct", ")")
        self.take("arrow")
        self.take("punct", "(")
        declared: list[TensorType] = []
        while not self.at("punct", ")"):
            if declared:
                self.take("punct", ",")
            declared.append(self.type())
        self.take("punct", ")")
        self.take("punct", "{")
        ops = []
        while not self.at("word", "return"):
            ops.append(self.op())
        ret = self.take("word", "return")
        outputs = [self.value_ref()]
        while self.at("punct", ","):
            self.pos += 1
            outputs.append(self.value_ref())
        self.take("punct", "}")
        self.take("eof")
        if [v.type for v in outputs] != declared:
            raise self.error("declared result types do not match the returned values", ret.span)
        return Func(name, args, ops, outputs)


def parse_func(text: str, filename: str = "<ir>", *, check: bool = True) -> Func:
    func = _Parser(SourceFile(filename, text)).func()
    if check:
        verify(func)
    return func
