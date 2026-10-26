"""`tinyc FILE.tiny`: compile through the pass pipeline; print IR, a schedule, kernels or a program."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from tinyc.driver import build_pipeline, compile_gpu, compile_tpu
from tinyc.frontend import CompileError, SourceFile, lower, parse
from tinyc.frontend.ast import dump
from tinyc.frontend.diagnostics import Diagnostic, Span
from tinyc.gpu.emit import dumps as dumps_launch
from tinyc.gpu.kernel_ir import print_kernel
from tinyc.ir import ref
from tinyc.ir.errors import PassError
from tinyc.ir.printer import print_func
from tinyc.ir.verifier import verify
from tinyc.passes import REGISTRY, PassContext, PassManager, PassVerifyError
from tinyc.tpu.emit import dumps
from tinyc.tpu.tile_ir import print_schedule

TPU_EMITS = ("tile", "ttpu")
GPU_EMITS = ("kernel", "tgpu", "launch")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tinyc", description=__doc__)
    ap.add_argument("file", type=Path)
    ap.add_argument("--emit", choices=["ast", "ir", *TPU_EMITS, *GPU_EMITS], default="ir",
                    help="tile/ttpu need --target tpu; kernel/tgpu/launch need --target gpu")  # fmt: skip
    ap.add_argument("--locs", action="store_true", help="print source locations in the IR")
    ap.add_argument("--target", choices=["gpu", "tpu"], help="also run that target's backend")
    ap.add_argument(
        "-O", dest="opt", type=int, choices=[0, 1], default=1, help="optimization level"
    )
    ap.add_argument("--disable", action="append", default=[], metavar="PASS",
                    help=f"skip a pass ({', '.join(REGISTRY)})")  # fmt: skip
    ap.add_argument("--print-after-all", action="store_true", help="dump the IR after every pass")
    ap.add_argument("--print-after", action="append", default=[], metavar="PASS")
    ap.add_argument("--verify-each", action="store_true",
                    help="run the program after every pass/stage and compare with the numpy oracle")  # fmt: skip
    ap.add_argument("--time-passes", action="store_true")
    ap.add_argument("--run", action="store_true",
                    help="run on the real chip/simulator with seeded random inputs")  # fmt: skip
    ap.add_argument("--seed", type=int, default=42, help="seed for --run inputs")
    ap.add_argument("--stats", action="store_true", help="print the static cost estimate")
    ap.add_argument("--gpu-regs", type=int, default=16, help="tiny-gpu registers per thread")
    args = ap.parse_args(argv)

    wants_backend = args.emit in (*TPU_EMITS, *GPU_EMITS) or args.run or args.stats
    if wants_backend and args.target is None:
        ap.error("--emit tile/ttpu/kernel/tgpu/launch, --run and --stats need --target gpu|tpu")
    if args.emit in TPU_EMITS and args.target != "tpu":
        ap.error(f"--emit {args.emit} needs --target tpu")
    if args.emit in GPU_EMITS and args.target != "gpu":
        ap.error(f"--emit {args.emit} needs --target gpu")
    source = SourceFile(args.file.name, args.file.read_text())

    def dump_ir(name: str, text: str) -> None:
        if args.print_after_all or name in args.print_after:
            print(f"// ---- after {name} ----\n{text}", file=sys.stderr, end="")

    ctx = PassContext(gpu_num_registers=args.gpu_regs)
    try:
        program = parse(source)
        if args.emit == "ast":
            print(dump(program.stmts))
            return 0
        if wants_backend:
            compile_fn = compile_tpu if args.target == "tpu" else compile_gpu
            kw = {"ctx": ctx} if args.target == "gpu" else {}
            comp = compile_fn(
                source.text, source.name, opt_level=args.opt, disabled=args.disable,
                verify_each=args.verify_each, dump=dump_ir, **kw,
            )  # fmt: skip
            if args.time_passes:
                print(comp.report.timing_table(), file=sys.stderr)
            return (_tpu_output if args.target == "tpu" else _gpu_output)(args, comp)
        func = lower(program, source)
        verify(func)
        manager = PassManager(
            build_pipeline(args.target, args.opt, args.disable),
            verify_each=args.verify_each,
            dump=dump_ir,
        )
        report = manager.run(func, ctx)
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


def _tpu_output(args, comp) -> int:
    est = comp.program["estimate"]
    if args.run:
        from runtime.tpu_runner import run_ttpu

        inputs = ref.random_inputs(comp.func, np.random.default_rng(args.seed))
        got = run_ttpu(comp.program, inputs)
        want = ref.evaluate(comp.func, inputs)
        exact = all((got.outputs[n] == w).all() for n, w in zip(comp.schedule.outputs, want))
        for name in comp.schedule.outputs:
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


def _gpu_output(args, comp) -> int:
    prog = comp.program
    est = prog.estimate
    if args.run:
        from runtime.gpu_runner import run_launch

        inputs = ref.random_inputs(comp.func, np.random.default_rng(args.seed))
        got = run_launch(comp.launch, inputs)
        want = ref.evaluate(comp.func, inputs)
        names = comp.launch["outputs"]
        exact = all((got.outputs[n] == w).all() for n, w in zip(names, want))
        for name in names:
            print(f"%{name} =\n{got.outputs[name]}")
        print(f"bit-exact vs numpy oracle: {'yes' if exact else 'NO'}")
        print(f"launches: {est['launches']}  threads: {est['threads']}  registers: {est['registers']}"
              f"  (R5-R7 untouched: {'yes' if got.reserved_intact else 'NO'})")  # fmt: skip
        c, i, o = est["cycles"], est["instructions"], got.op_counts
        print(f"measured cycles: {got.cycles}  (estimate {c['min']}..{c['max']})")
        print(f"measured instructions: {got.instructions}  (estimate {i['min']}..{i['max']})")
        print(f"measured loads: {o['LD']} (estimate {est['loads']})  "
              f"stores: {o['ST']} (estimate {est['stores']})")  # fmt: skip
        print("note: tiny-gpu runs straight-line code in one cycle, so compare instructions too")
        return 0 if exact and got.reserved_intact else 1
    if args.stats:
        print(json.dumps(est, indent=2))
    elif args.emit == "kernel":
        print("\n".join(print_kernel(k) for k in prog.kernels), end="")
    elif args.emit == "tgpu":
        print(prog.asm_text(), end="")
    else:
        print(dumps_launch(comp.launch), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
