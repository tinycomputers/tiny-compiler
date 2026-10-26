"""`tinyc FILE.tiny [--emit ast|ir] [--locs]`: print the AST or Graph IR, or a caret diagnostic."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tinyc.frontend import CompileError, SourceFile, lower, parse
from tinyc.frontend.ast import dump
from tinyc.ir.printer import print_func
from tinyc.ir.verifier import verify


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tinyc", description=__doc__)
    ap.add_argument("file", type=Path)
    ap.add_argument("--emit", choices=["ast", "ir"], default="ir")
    ap.add_argument("--locs", action="store_true", help="print source locations in the IR")
    args = ap.parse_args(argv)
    source = SourceFile(args.file.name, args.file.read_text())
    try:
        program = parse(source)
        if args.emit == "ast":
            print(dump(program.stmts))
            return 0
        func = lower(program, source)
        verify(func)
    except CompileError as err:
        print(err.render(), file=sys.stderr)
        return 1
    print(print_func(func, locs=args.locs), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
