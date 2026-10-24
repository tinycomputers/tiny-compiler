"""`tinyc FILE.tiny`: compile to Graph IR (optionally through the pass pipeline) and print it."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tinyc.driver import build_pipeline
from tinyc.frontend import CompileError, SourceFile, lower, parse
from tinyc.frontend.ast import dump
from tinyc.frontend.diagnostics import Diagnostic, Span
from tinyc.ir.errors import PassError
from tinyc.ir.printer import print_func
from tinyc.ir.verifier import verify
from tinyc.passes import REGISTRY, PassManager, PassVerifyError


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tinyc", description=__doc__)
    ap.add_argument("file", type=Path)
    ap.add_argument("--emit", choices=["ast", "ir"], default="ir")
    ap.add_argument("--locs", action="store_true", help="print source locations in the IR")
    ap.add_argument("--target", choices=["gpu", "tpu"], help="also run that target's legality pass")
    ap.add_argument(
        "-O", dest="opt", type=int, choices=[0, 1], default=1, help="optimization level"
    )
    ap.add_argument("--disable", action="append", default=[], metavar="PASS",
                    help=f"skip a pass ({', '.join(REGISTRY)})")  # fmt: skip
    ap.add_argument("--print-after-all", action="store_true", help="dump the IR after every pass")
    ap.add_argument("--print-after", action="append", default=[], metavar="PASS")
    ap.add_argument("--verify-each", action="store_true",
                    help="run the IR after every pass and compare with the numpy oracle")  # fmt: skip
    ap.add_argument("--time-passes", action="store_true")
    args = ap.parse_args(argv)

    source = SourceFile(args.file.name, args.file.read_text())

    def dump_ir(name: str, text: str) -> None:
        if args.print_after_all or name in args.print_after:
            print(f"// ---- after {name} ----\n{text}", file=sys.stderr, end="")

    try:
        program = parse(source)
        if args.emit == "ast":
            print(dump(program.stmts))
            return 0
        func = lower(program, source)
        verify(func)
        manager = PassManager(
            build_pipeline(args.target, args.opt, args.disable),
            verify_each=args.verify_each,
            dump=dump_ir,
        )
        report = manager.run(func)
    except CompileError as err:
        print(err.render(), file=sys.stderr)
        return 1
    except PassError as err:
        if err.loc is None:
            print(f"{source.name}: error: {err.message}", file=sys.stderr)
        else:
            span = Span(err.loc.line, err.loc.col, err.loc.line, err.loc.col + 1)
            print(CompileError(source, Diagnostic(err.message, span)).render(), file=sys.stderr)
        return 1
    except PassVerifyError as err:
        print(f"internal compiler error: {err}", file=sys.stderr)
        return 3
    except ValueError as err:  # unknown --disable name
        ap.error(str(err))
    if args.time_passes:
        print(report.timing_table(), file=sys.stderr)
    print(print_func(func, locs=args.locs), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
