"""The measured optimization table in docs/ARCHITECTURE.md, generated from real simulator runs.

`python scripts/opt_table.py --write` regenerates it; a test fails if the document drifts.
"""

from __future__ import annotations

import numpy as np

from tinyc.driver import compile_gpu
from tinyc.ir import ref

PROGRAMS = {
    "4x4": "input A : i8[4, 4]\ninput B : i8[4, 4]\nC = relu(A @ B)\noutput C\n",
    "8x8": "input A : i8[8, 8]\ninput B : i8[8, 8]\nC = relu(A @ B)\noutput C\n",
}

# (label, -O level, kernel passes switched off)
CONFIGS = [
    ("-O0 (naive)", 0, ()),
    ("+ licm", 1, ("strength-reduce",)),
    ("+ strength-reduce (= -O1)", 1, ()),
    ("-O2 with shared-memory tiling only", 2, ("unroll", "fold")),
    ("-O2 with unrolling + folding only", 2, ("shared-tile",)),
    ("-O2 (all passes)", 2, ()),
]
SEED = 42


def measure(source: str, opt: int, disabled=()) -> dict:
    from runtime.gpu_runner import run_launch

    comp = compile_gpu(source, "mm_relu.tiny", opt_level=opt, disabled=disabled)
    inputs = ref.random_inputs(comp.func, np.random.default_rng(SEED))
    res = run_launch(comp.launch, inputs)
    want = ref.evaluate(comp.func, inputs)
    exact = all((res.outputs[n] == w).all() for n, w in zip(comp.launch["outputs"], want))
    assert exact and res.reserved_intact, "a measured configuration must be correct"
    ops = res.op_counts
    return {
        "instructions": res.instructions,
        "cycles": res.cycles,
        "loads": ops["LD"],
        "shared": ops["SHLD"],
        "registers": comp.program.estimate["registers"],
    }


def opt_table_markdown() -> str:
    out: list[str] = []
    for name, source in PROGRAMS.items():
        rows = [(label, measure(source, opt, off)) for label, opt, off in CONFIGS]
        base = rows[0][1]
        out.append(f"**`mm_relu` {name}** (`C = relu(A @ B)`, int8, seed {SEED})\n")
        out.append(
            "| configuration | instructions | cycles | global loads | shared loads | registers |"
        )
        out.append("|---|---:|---:|---:|---:|---:|")
        for label, m in rows:
            cells = [
                (
                    f"{m['instructions']} ({m['instructions'] / base['instructions'] - 1:+.0%})"
                    if m is not base
                    else str(m["instructions"])
                ),
                str(m["cycles"]),
                (
                    f"{m['loads']} ({m['loads'] / base['loads'] - 1:+.0%})"
                    if m is not base
                    else str(m["loads"])
                ),
                str(m["shared"]),
                str(m["registers"]),
            ]
            out.append(f"| {label} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)
