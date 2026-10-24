"""The real chip: every program runs on the Verilated tiny-tpu RTL through the JS executor."""

import numpy as np
import pytest

from runtime.tpu_runner import run_tpu_passes, run_ttpu
from tests.tpu.helpers import TPU_EXAMPLES, ZERO_K_TILE, build
from tinyc.ir import ref
from tinyc.tpu.tile_ir import CYCLES_PER_PASS


def test_the_cost_models_pass_cost_is_what_the_rtl_measures():
    eye = np.eye(4, dtype=np.int64)
    assert run_tpu_passes([(eye, eye)]).passes[0].cycles == CYCLES_PER_PASS


@pytest.mark.parametrize("path", TPU_EXAMPLES, ids=lambda p: p.stem)
@pytest.mark.parametrize("opt", [0, 1])
def test_examples_are_bit_exact_and_the_cost_model_equals_the_measurement(path, opt):
    comp = build(path.read_text(), opt=opt)
    est = comp.program["estimate"]
    for seed in (1, 2):
        inputs = ref.random_inputs(comp.func, np.random.default_rng(seed))
        got = run_ttpu(comp.program, inputs)
        for name, want in zip(comp.schedule.outputs, ref.evaluate(comp.func, inputs)):
            np.testing.assert_array_equal(got.outputs[name], want)
        assert len(got.pass_cycles) == est["passes"]
        assert set(got.pass_cycles) <= {CYCLES_PER_PASS}
        assert got.total_cycles == est["rtl_cycles"]  # estimate == measurement, exactly
        assert got.host_ops == est["host_ops"]


def test_big_matmul_is_27_passes_on_the_chip():
    comp = build((TPU_EXAMPLES[0].parent / "big_mm.tiny").read_text())
    inputs = ref.random_inputs(comp.func, np.random.default_rng(5))
    got = run_ttpu(comp.program, inputs)
    assert len(got.pass_cycles) == 27 and got.total_cycles == 405
    assert got.host_ops["acc"] == 27  # 27 accumulations, all done by the host


def test_skipping_a_zero_tile_saves_exactly_one_passes_worth_of_cycles():
    inputs = {"X": np.arange(-16, 16).reshape(4, 8)}
    slow = run_ttpu(build(ZERO_K_TILE, opt=0).program, inputs)
    fast = run_ttpu(build(ZERO_K_TILE, opt=1).program, inputs)
    np.testing.assert_array_equal(fast.outputs["2"], slow.outputs["2"])
    assert slow.total_cycles - fast.total_cycles == CYCLES_PER_PASS
    assert fast.total_cycles == 15


def test_partial_tiles_are_zero_padded_correctly():
    comp = build("input A : i8[5, 6]\ninput B : i8[6, 7]\nC = relu(A @ B)\noutput C\n")
    rng = np.random.default_rng(11)
    inputs = {"A": rng.integers(-128, 128, (5, 6)), "B": rng.integers(-128, 128, (6, 7))}
    got = run_ttpu(comp.program, inputs)
    np.testing.assert_array_equal(got.outputs["1"], np.maximum(inputs["A"] @ inputs["B"], 0))
    assert len(got.pass_cycles) == 8


def test_extreme_values_through_a_full_program():
    comp = build("input A : i8[4, 8]\ninput B : i8[8, 4]\nC = A @ B\noutput C\n")
    inputs = {"A": np.full((4, 8), -128), "B": np.full((8, 4), -128)}
    got = run_ttpu(comp.program, inputs)
    assert (got.outputs["0"] == 8 * 128 * 128).all()


def test_the_executor_rejects_bad_inputs():
    comp = build("input A : i8[4, 4]\ninput B : i8[4, 4]\nC = A @ B\noutput C\n")
    ok = {"A": np.zeros((4, 4), dtype=int), "B": np.zeros((4, 4), dtype=int)}
    run_ttpu(comp.program, ok)
    with pytest.raises(RuntimeError, match="missing input 'B'"):
        run_ttpu(comp.program, {"A": ok["A"]})
    with pytest.raises(RuntimeError, match="values outside i8"):
        run_ttpu(comp.program, {**ok, "A": np.full((4, 4), 200)})
    with pytest.raises(RuntimeError, match="expected shape"):
        run_ttpu(comp.program, {**ok, "B": np.zeros((4, 5), dtype=int)})
