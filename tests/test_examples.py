"""Every file in examples/ parses, type-checks, round-trips and agrees across both evaluators."""

from pathlib import Path

import numpy as np
import pytest

from tinyc.frontend import compile_to_graph
from tinyc.ir import interp, ref
from tinyc.ir.parser import parse_func
from tinyc.ir.printer import print_func

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = sorted((ROOT / "examples").glob("*.tiny"))
IDS = [p.stem for p in EXAMPLES]


def load(path):
    return compile_to_graph(path.read_text(), path.name)


def test_examples_exist():
    assert {"mm", "mm_relu", "mm_bias_relu", "const_mm", "mlp_gpu", "big_mm"} <= set(IDS)


@pytest.mark.parametrize("path", EXAMPLES, ids=IDS)
def test_ir_golden_and_round_trip(path, golden):
    f = load(path)
    text = print_func(f, locs=True)
    golden(ROOT / "tests" / "golden" / "ir" / f"{path.stem}.ir", text)
    assert print_func(parse_func(text), locs=True) == text


@pytest.mark.parametrize("path", EXAMPLES, ids=IDS)
def test_interp_equals_oracle(path, rng):
    f = load(path)
    for _ in range(10):
        inputs = ref.random_inputs(f, rng)
        for got, want in zip(interp.run(f, inputs), ref.evaluate(f, inputs)):
            np.testing.assert_array_equal(got, want)


def test_const_example_by_hand():
    f = load(ROOT / "examples" / "const_mm.tiny")
    (out,) = ref.evaluate(f, {"X": [[1, 1], [2, 0], [-1, 0]]})
    # W = [[1,-2],[3,4]]: rows -> [4,2], [2,-4], [-1,2]; relu clamps negatives
    assert out.tolist() == [[4, 2], [2, 0], [0, 2]]
