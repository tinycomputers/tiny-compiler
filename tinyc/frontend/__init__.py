"""`.tiny` source -> Graph IR."""

from tinyc.frontend.diagnostics import CompileError, SourceFile
from tinyc.frontend.lower import lower
from tinyc.frontend.parser import parse
from tinyc.ir.graph import Func
from tinyc.ir.verifier import verify


def compile_to_graph(text: str, filename: str = "<input>") -> Func:
    """Parse, type-check and verify. Raises CompileError with a caret-style message."""
    source = SourceFile(filename, text)
    func = lower(parse(source), source)
    verify(func)
    return func


__all__ = ["CompileError", "SourceFile", "compile_to_graph", "lower", "parse"]
