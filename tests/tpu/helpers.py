from __future__ import annotations

from pathlib import Path

import numpy as np

from tinyc.driver import compile_tpu
from tinyc.ir import ref
from tinyc.tpu import interp

ROOT = Path(__file__).resolve().parent.parent.parent
TPU_EXAMPLES = sorted(p for p in (ROOT / "examples").glob("*.tiny") if p.stem != "mlp_gpu")

# W's second K tile (rows 4..8) is all zero: that pass contributes nothing.
ZERO_K_TILE = """\
input X : i8[4, 8]
const W : i8[8, 4] = [[1, 2, 3, 4], [5, 6, 7, 8], [1, 1, 1, 1], [2, 2, 2, 2],
                      [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
Y = relu(X @ W)
output Y
"""


def check_against_oracle(comp, seeds=(1, 2, 3)):
    """Python reference interpreter on the emitted JSON vs the numpy oracle."""
    for seed in seeds:
        inputs = ref.random_inputs(comp.func, np.random.default_rng(seed))
        got = interp.run_program(comp.program, inputs)["outputs"]
        for name, want in zip(comp.schedule.outputs, ref.evaluate(comp.func, inputs)):
            np.testing.assert_array_equal(got[name], want)


def build(source, opt=1, **kw):
    return compile_tpu(source, "t.tiny", opt_level=opt, **kw)
