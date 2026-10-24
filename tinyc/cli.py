"""`tinyc FILE.tiny`: compile through the pass pipeline; print IR, a tile schedule or a program."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from tinyc.driver import build_pipeline, compile_tpu
from tinyc.frontend import CompileError, SourceFile, lower, parse
from tinyc.frontend.ast import dump
from tinyc.frontend.diagnostics import Diagnostic, Span
from tinyc.ir import ref
from tinyc.ir.errors import PassError
from tinyc.ir.printer import print_func
from tinyc.ir.verifier import verify
from tinyc.passes import REGISTRY, PassManager, PassVerifyError
from tinyc.tpu.emit import dumps
from tinyc.tpu.tile_ir import print_schedule


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tinyc", description=__doc__)
    ap.add_argument("file", type=Path)
    ap.add_argument("--emit", choices=["ast", "ir", "tile", "ttpu"], default="ir",
                    help="tile and ttpu need --target tpu")  # fmt: skip
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
    ap.add_argument("--run", action="store_true",
                    help="run on the real chip with seeded random inputs (--target tpu)")  # fmt: skip
    ap.add_argument("--seed", type=int, default=42, help="seed for --run inputs")
    ap.add_argument("--stats", action="store_true", help="print the static cost estimate")
    args = ap.parse_args(argv)

    if (args.emit in ("tile", "ttpu") or args.run or args.stats) and args.target != "tpu":
        ap.error(
            "--emit tile/ttpu, --run and --stats need --target tpu (the gpu backend comes later)"
        )
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
        if args.target == "tpu" and (args.emit in ("tile", "ttpu") or args.run or args.stats):
            # the backend re-runs the graph pipeline itself, so the dumps above are not repeated
            comp = compile_tpu(
                source.text, source.name, opt_level=args.opt, disabled=args.disable,
                verify_each=args.verify_each, dump=dump_ir if args.print_after_all else None,
            )  # fmt: skip
            return _tpu_output(args, comp, report)
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


def _tpu_output(args, comp, report) -> int:
    est = comp.program["estimate"]
    if args.time_passes:
        print(report.timing_table(), file=sys.stderr)
    if args.run:
        from runtime.tpu_runner import run_ttpu

        inputs = ref.random_inputs(comp.func, np.random.default_rng(args.seed))
        got = run_ttpu(comp.program, inputs)
        want = ref.evaluate(comp.func, inputs)
        exact = all((got.outputs[name] == w).all() for name, w in zip(comp.schedule.outputs, want))
        for name, w in zip(comp.schedule.outputs, want):
            print(f"%{name} =\n{got.outputs[name]}")
        print(f"bit-exact vs numpy oracle: {'yes' if exact else 'NO'}")
        print(f"chip passes: {len(got.pass_cycles)}  measured RTL cycles: {got.total_cycles}"
              f"  (estimate {est['rtl_cycles']})")  # fmt: skip
        print(f"host ops (not chip work): {got.host_ops}")
        return 0 if exact else 1
    if args.stats:
        print(json.dumps(est, indent=2))
    elif args.emit == "tile":
        print(print_schedule(comp.schedule), end="")
    else:
        print(dumps(comp.program), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
