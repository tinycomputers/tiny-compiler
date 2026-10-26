"""Lexer for `.tiny`. Newlines end statements, except inside (...) and [...]."""

from __future__ import annotations

import re
from dataclasses import dataclass

from tinyc.frontend.diagnostics import CompileError, Diagnostic, SourceFile, Span

KEYWORDS = {"input", "const", "output", "relu", "i8", "i32"}

_TOKEN = re.compile(
    r"(?P<ws>[ \t\r]+)|(?P<comment>\#[^\n]*)|(?P<nl>\n)"
    r"|(?P<name>[A-Za-z_][A-Za-z0-9_]*)|(?P<int>[0-9]+)|(?P<punct>[:=@+()\[\],-])"
)


@dataclass(frozen=True)
class Token:
    kind: str  # NAME | INT | NEWLINE | EOF | a keyword | a punctuation character
    text: str
    span: Span


def lex(source: SourceFile) -> list[Token]:
    text = source.text
    tokens: list[Token] = []
    pos, line, line_start, depth = 0, 1, 0, 0

    def span(start: int, end: int) -> Span:
        return Span(line, start - line_start + 1, line, end - line_start + 1)

    def newline(start: int) -> None:
        if depth == 0 and tokens and tokens[-1].kind != "NEWLINE":
            tokens.append(Token("NEWLINE", "\n", span(start, start + 1)))

    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if m is None:
            bad = Diagnostic(f"unexpected character {text[pos]!r}", span(pos, pos + 1))
            raise CompileError(source, bad)
        kind, lexeme = m.lastgroup, m.group()
        if kind == "nl":
            newline(pos)
            line, line_start = line + 1, m.end()
        elif kind == "name":
            tokens.append(
                Token(lexeme if lexeme in KEYWORDS else "NAME", lexeme, span(pos, m.end()))
            )
        elif kind == "int":
            tokens.append(Token("INT", lexeme, span(pos, m.end())))
        elif kind == "punct":
            depth = max(depth + {"(": 1, "[": 1, ")": -1, "]": -1}.get(lexeme, 0), 0)
            tokens.append(Token(lexeme, lexeme, span(pos, m.end())))
        pos = m.end()

    newline(pos)
    tokens.append(Token("EOF", "", span(pos, pos)))
    return tokens
