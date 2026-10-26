"""Prove every value fits in i32 by interval analysis, or reject the program.

tiny-gpu raises on i32 overflow and tiny-tpu accumulates in 32 bits, so the compiler must show
no intermediate can overflow for *any* legal input. Bounds:
  * i8 inputs are exactly [-128, 127]; i32 inputs are assumed to lie in `ctx.i32_input_range`;
    constants are exact;
  * a matmul sum of K products, each within [pmin, pmax], lies in [K*pmin, K*pmax]; every
    *partial* sum lies in [min(0, K*pmin), max(0, K*pmax)], and that envelope must fit too,
    so the proof holds for any accumulation order (tiling on the TPU, the loop on the GPU).
Results go to `ctx.analysis["ranges"]` as {value name: (lo, hi)}.
"""

from __future__ import annotations

from tinyc.ir.errors import PassError
from tinyc.ir.types import RANGES, epilogue

I32_LO, I32_HI = RANGES["i32"]
Interval = tuple[int, int]


def _mul(a: Interval, b: Interval) -> Interval:
    corners = [x * y for x in a for y in b]
    return min(corners), max(corners)


def _need_fit(op, lo: int, hi: int, what: str, ctx, has_i32_input: bool) -> None:
    if I32_LO <= lo and hi <= I32_HI:
        return
    note = (
        f" (i32 inputs are assumed to lie in {list(ctx.i32_input_range)})" if has_i32_input else ""
    )
    raise PassError(
        f"%{op.result.name} ({what}) may reach {lo}..{hi}, which does not fit in i32{note}",
        op.loc,
    )


class RangeCheck:
    name = "range-check"

    def run(self, func, ctx):
        ranges: dict[str, Interval] = {}
        for a in func.args:
            ranges[a.name] = RANGES["i8"] if a.type.dtype == "i8" else ctx.i32_input_range
        has_i32 = any(a.type.dtype == "i32" for a in func.args)

        for op in func.ops:
            xs = [ranges[v.name] for v in op.operands]

            def fit(r: Interval, what: str, op=op) -> None:
                _need_fit(op, *r, what, ctx, has_i32)

            if op.kind == "const":
                r = (int(op.data.min()), int(op.data.max()))
            elif op.kind == "relu":
                r = (max(xs[0][0], 0), max(xs[0][1], 0))
            elif op.kind == "add":
                r = (xs[0][0] + xs[1][0], xs[0][1] + xs[1][1])
                fit(r, "add")
            else:  # matmul, possibly fused
                k = op.operands[0].type.shape[1]
                pmin, pmax = _mul(xs[0], xs[1])
                fit((min(0, k * pmin), max(0, k * pmax)), "matmul partial sums")
                r = (k * pmin, k * pmax)
                for stage in epilogue(op.attrs):
                    if stage == "bias":
                        r = (r[0] + xs[2][0], r[1] + xs[2][1])
                        fit(r, "matmul bias add")
                    else:
                        r = (max(r[0], 0), max(r[1], 0))
            ranges[op.result.name] = r
        ctx.analysis["ranges"] = ranges
