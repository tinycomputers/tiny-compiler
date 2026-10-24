"""Hypothesis strategy for random legal `.tiny` programs (small dims, so ops often line up)."""

from __future__ import annotations

from hypothesis import strategies as st

from tinyc.ir.types import RANGES

DIMS = st.sampled_from([1, 2, 3, 4])


def _fmt(dtype, shape):
    return f"{dtype}[{', '.join(map(str, shape))}]"


def _nested(draw, shape, lo, hi):
    if len(shape) == 1:  # zeros are common on purpose: they make whole constant tiles zero
        return [draw(st.one_of(st.just(0), st.integers(lo, hi))) for _ in range(shape[0])]
    return [_nested(draw, shape[1:], lo, hi) for _ in range(shape[0])]


@st.composite
def programs(draw, max_ops: int = 6) -> str:
    pool: list[tuple[str, str, tuple[int, ...]]] = []  # (name, dtype, shape)
    lines: list[str] = []
    for i in range(draw(st.integers(2, 4))):
        shape = tuple(draw(DIMS) for _ in range(draw(st.sampled_from([1, 2, 2]))))
        dtype = draw(st.sampled_from(["i8", "i8", "i32"]))
        lines.append(f"input x{i} : {_fmt(dtype, shape)}")
        pool.append((f"x{i}", dtype, shape))
    for i in range(draw(st.integers(0, 2))):
        shape = tuple(draw(DIMS) for _ in range(draw(st.sampled_from([1, 2]))))
        lo, hi = RANGES["i8"]
        lines.append(f"const c{i} : {_fmt('i8', shape)} = {_nested(draw, shape, lo, hi)}")
        pool.append((f"c{i}", "i8", shape))

    nbias = 0
    for j in range(draw(st.integers(1, max_ops))):
        matmuls = [
            (a, b)
            for a in pool
            for b in pool
            if len(a[2]) == 2 and len(b[2]) == 2 and a[2][1] == b[2][0]
        ]
        adds = []
        for a in pool:
            for b in pool:
                same = a[2] == b[2]
                bcast = len(a[2]) == 2 and len(b[2]) == 1 and a[2][1] == b[2][0]
                if same or bcast or (len(b[2]) == 2 and len(a[2]) == 1 and b[2][1] == a[2][0]):
                    adds.append((a, b))
        kinds = ["relu"] + (["add"] if adds else []) + (["matmul", "dense"] if matmuls else [])
        kind = kinds[draw(st.integers(0, len(kinds) - 1))]
        if kind == "relu":
            v = pool[draw(st.integers(0, len(pool) - 1))]
            expr, dtype, shape = f"relu({v[0]})", v[1], v[2]
        elif kind == "add":
            a, b = adds[draw(st.integers(0, len(adds) - 1))]
            expr, dtype = f"{a[0]} + {b[0]}", "i32"
            shape = a[2] if len(a[2]) >= len(b[2]) else b[2]
        else:
            a, b = matmuls[draw(st.integers(0, len(matmuls) - 1))]
            dtype, shape = "i32", (a[2][0], b[2][1])
            if kind == "matmul":
                expr = f"{a[0]} @ {b[0]}"
            else:  # a dense-layer idiom, so the fuser has something to fuse
                form = draw(
                    st.sampled_from(["relu({m})", "{m} + {w}", "relu({m} + {w})", "{w} + {m}"])
                )
                m = f"{a[0]} @ {b[0]}"
                if "{w}" in form:
                    bias = f"w{nbias}"
                    nbias += 1
                    bdtype = draw(st.sampled_from(["i8", "i32"]))
                    lines.append(f"input {bias} : {_fmt(bdtype, (shape[1],))}")
                    pool.append((bias, bdtype, (shape[1],)))
                else:
                    bias = ""
                expr = form.format(m=m, w=bias)
        lines.append(f"v{j} = {expr}")
        pool.append((f"v{j}", dtype, shape))

    outs = [pool[-1][0]]
    extra = draw(st.integers(0, len(pool) - 1))
    if pool[extra][0] not in outs and draw(st.booleans()):
        outs.append(pool[extra][0])
    lines.append("output " + ", ".join(outs))
    return "\n".join(lines) + "\n"


TPU_DIMS = st.sampled_from([1, 2, 3, 4, 5, 8, 9])  # 1..3 tiles per dimension, some partial
EPILOGUES = [
    "{m}",
    "relu({m})",
    "{m} + {w}",
    "relu({m} + {w})",
    "{w} + {m}",
    "relu(relu({m}))",
    "{m} + {m}",
    "relu({m}) + relu({m})",
    "relu({m} + {w}) + {m}",
]


def _const_with_zero_blocks(draw, rows, cols):
    """An i8 matrix whose 4x4 blocks are zero with probability 0.4 (so whole tiles skip)."""
    lo, hi = RANGES["i8"]
    zero = {
        (bi, bj): draw(st.integers(0, 4)) < 2
        for bi in range((rows + 3) // 4)
        for bj in range((cols + 3) // 4)
    }
    return [
        [0 if zero[(i // 4, j // 4)] else draw(st.integers(lo, hi)) for j in range(cols)]
        for i in range(rows)
    ]


@st.composite
def tpu_programs(draw) -> str:
    """Programs aimed at tiny-tpu: multi-tile matmuls, constants with zero tiles, every epilogue
    shape, and a mix of fused and host-side add/relu. Always legal for the TPU."""
    lines: list[str] = []
    outs: list[str] = []
    for chain in range(draw(st.integers(1, 2))):
        m, k, n = draw(TPU_DIMS), draw(TPU_DIMS), draw(TPU_DIMS)
        left_const = draw(st.integers(0, 4)) == 0
        right_const = not left_const and draw(st.booleans())
        if left_const:
            lines.append(f"const L{chain} : i8[{m}, {k}] = {_const_with_zero_blocks(draw, m, k)}")
            lines.append(f"input R{chain} : i8[{k}, {n}]")
            a, b = f"L{chain}", f"R{chain}"
        elif right_const:
            lines.append(f"input L{chain} : i8[{m}, {k}]")
            lines.append(f"const R{chain} : i8[{k}, {n}] = {_const_with_zero_blocks(draw, k, n)}")
            a, b = f"L{chain}", f"R{chain}"
        else:
            lines.append(f"input L{chain} : i8[{m}, {k}]")
            lines.append(f"input R{chain} : i8[{k}, {n}]")
            a, b = f"L{chain}", f"R{chain}"
        form = draw(st.sampled_from(EPILOGUES))
        if "{w}" in form:
            lines.append(f"input w{chain} : {draw(st.sampled_from(['i8', 'i32']))}[{n}]")
        lines.append(f"Y{chain} = " + form.format(m=f"{a} @ {b}", w=f"w{chain}"))
        outs.append(f"Y{chain}")
    lines.append("output " + ", ".join(outs))
    return "\n".join(lines) + "\n"
