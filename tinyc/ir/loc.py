"""Source locations. Every op carries one so any lowered line can be mapped back to the source."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Loc:
    file: str
    line: int
    col: int

    def __str__(self) -> str:
        return f'loc("{self.file}":{self.line}:{self.col})'
