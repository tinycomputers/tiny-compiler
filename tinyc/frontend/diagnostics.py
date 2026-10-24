"""Source spans and caret-style error rendering. Error messages are a feature (PLAN §3.2)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Span:
    """1-based, end column exclusive."""

    line: int
    col: int
    end_line: int
    end_col: int

    @staticmethod
    def join(a: Span, b: Span) -> Span:
        return Span(a.line, a.col, b.end_line, b.end_col)


class SourceFile:
    def __init__(self, name: str, text: str):
        self.name = name
        self.text = text
        self.lines = text.split("\n")

    def line_text(self, n: int) -> str:
        return self.lines[n - 1] if 1 <= n <= len(self.lines) else ""


@dataclass(frozen=True)
class Diagnostic:
    message: str
    span: Span
    note: str | None = None


class CompileError(Exception):
    def __init__(self, source: SourceFile, diagnostic: Diagnostic):
        self.source = source
        self.diagnostic = diagnostic
        super().__init__(self.render())

    def render(self) -> str:
        d, s = self.diagnostic, self.diagnostic.span
        text = self.source.line_text(s.line)
        width = (s.end_col - s.col) if s.end_line == s.line else (len(text) - s.col + 1)
        gutter = len(str(s.line))
        pad = "".join(ch if ch == "\t" else " " for ch in text[: s.col - 1])
        out = [
            f"{self.source.name}:{s.line}:{s.col}: error: {d.message}",
            f"{s.line} | {text}",
            f"{' ' * gutter} | {pad}{'^' * max(width, 1)}",
        ]
        if d.note:
            out.append(f"{' ' * gutter} = note: {d.note}")
        return "\n".join(out)
