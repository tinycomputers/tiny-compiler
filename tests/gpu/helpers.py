from __future__ import annotations

from pathlib import Path

import numpy as np

from tinyc.driver import compile_gpu
from tinyc.ir import ref

ROOT = Path(__file__).resolve().parent.parent.parent
EXAMPLES = sorted((ROOT / "examples").glob("*.tiny"))  # every example is legal on tiny-gpu


def build(source: str, opt: int = 1, regs: int = 16, **kw):
    from tinyc.passes import PassContext

    return compile_gpu(
        source, "t.tiny", opt_level=opt, ctx=PassContext(gpu_num_registers=regs), **kw
    )


def run_on_simulator(comp, seed: int = 1, inputs=None):
    """Run on the real tiny-gpu simulator. Returns (LaunchResult, inputs, oracle outputs)."""
    from runtime.gpu_runner import run_launch

    inputs = inputs or ref.random_inputs(comp.func, np.random.default_rng(seed))
    return run_launch(comp.launch, inputs), inputs, ref.evaluate(comp.func, inputs)


def assert_bit_exact(comp, result, want):
    for name, w in zip(comp.launch["outputs"], want):
        np.testing.assert_array_equal(result.outputs[name], w)
