"""Shared test helpers: run named passes over IR text and compare printed results."""

from __future__ import annotations

import textwrap

from tinyc.ir.parser import parse_func
from tinyc.ir.printer import print_func
from tinyc.passes import REGISTRY, PassContext, PassManager


def ir(text: str):
    return parse_func(textwrap.dedent(text).strip() + "\n")


def run_passes(text: str, *names: str, ctx: PassContext | None = None) -> str:
    """Parse `text`, run the named passes with translation validation on, return printed IR."""
    func = ir(text)
    PassManager([REGISTRY[n]() for n in names], verify_each=True).run(func, ctx)
    return print_func(func)


def expected(text: str) -> str:
    return textwrap.dedent(text).strip() + "\n"
