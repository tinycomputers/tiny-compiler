"""The real thing: every program runs on the unmodified tiny-gpu simulator."""

import numpy as np
import pytest

from runtime.gpu_runner import run_launch
from runtime.tpu_runner import run_ttpu
from tests.gpu.helpers import EXAMPLES, ROOT, assert_bit_exact, build, run_on_simulator
from tinyc.driver import compile_tpu
from tinyc.gpu import interp
from tinyc.ir import ref
from tinyc.ir.errors import PassError

MM_RELU = "input A : i8[4, 8]\ninput B : i8[8, 4]\nC = relu(A @ B)\noutput C\n"


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
@pytest.mark.parametrize("opt", [0, 1, 2])
def test_examples_are_bit_exact_and_never_touch_the_reserved_registers(path, opt):
    comp = build(path.read_text(), opt=opt)
    for seed in (1, 2):
        res, _, want = run_on_simulator(comp, seed)
        assert_bit_exact(comp, res, want)
        assert res.reserved_intact


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
@pytest.mark.parametrize("opt", [0, 1, 2])
def test_estimate_bounds_contain_the_measurement_and_memory_ops_are_exact(path, opt):
    comp = build(path.read_text(), opt=opt)
    est = comp.program.estimate
    res, _, _ = run_on_simulator(comp, 7)
    assert est["instructions"]["min"] <= res.instructions <= est["instructions"]["max"]
    assert est["cycles"]["min"] <= res.cycles <= est["cycles"]["max"]
    assert res.op_counts["LD"] == est["loads"] and res.op_counts["ST"] == est["stores"]
    assert res.op_counts["SHLD"] == est["shared_loads"]
    assert res.op_counts["SHST"] == est["shared_stores"]
    for (name, r), per in zip(res.kernels, est["per_kernel"]):
        assert name == per["name"]
        assert per["cycles"]["min"] <= r.cycles <= per["cycles"]["max"]


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
@pytest.mark.parametrize("opt", [0, 1, 2])
def test_interpreter_counts_equal_simulator_counts_exactly(path, opt):
    """Checks lowering, register allocation and emission together: the structured Kernel IR,
    interpreted without any registers, must execute exactly what the simulator executes."""
    comp = build(path.read_text(), opt=opt)
    res, inputs, _ = run_on_simulator(comp, 3)
    got = interp.run_program(comp.program, inputs)
    for (name, sim), counts in zip(res.kernels, got["kernels"]):
        assert name == counts.name
        assert dict(sim.op_counts) == dict(counts.ops)
        assert (sim.instructions, sim.cycles) == (counts.instructions, counts.cycles)
    for n in comp.launch["outputs"]:
        np.testing.assert_array_equal(got["outputs"][n], res.outputs[n])


@pytest.mark.parametrize("opt", [0, 1, 2])
def test_the_estimate_bounds_are_attained_at_the_data_extremes(opt):
    comp = build(MM_RELU, opt=opt)
    est = comp.program.estimate
    ones = np.ones((4, 8), dtype=np.int64)
    # every product positive: nothing clamps -> fewest instructions, but every thread takes the skip branch
    keep = run_launch(comp.launch, {"A": ones, "B": np.ones((8, 4), dtype=np.int64)})
    assert keep.instructions == est["instructions"]["min"] and keep.cycles == est["cycles"]["max"]
    # every product negative: everything clamps -> one more instruction per thread, one cycle fewer
    clamp = run_launch(comp.launch, {"A": ones, "B": -np.ones((8, 4), dtype=np.int64)})
    assert clamp.instructions == est["instructions"]["max"] and clamp.cycles == est["cycles"]["min"]
    assert (clamp.outputs["1"] == 0).all()


def test_a_straight_line_kernel_is_one_cycle_however_many_instructions():
    comp = build("input a : i8[8]\ninput b : i8[8]\nC = a + b\noutput C\n")
    res, _, _ = run_on_simulator(comp)
    assert res.cycles == 1 and res.instructions == 8 * 7  # 7 instructions per thread, 8 threads


@pytest.mark.parametrize("opt", [0, 1, 2])
def test_without_data_dependent_branches_the_estimate_is_exact_at_every_level(opt):
    """No relu -> nothing is uncertain: instructions and cycles must equal the measurement,
    including the barrier model of the tiled -O2 kernel."""
    comp = build("input A : i8[4, 8]\ninput B : i8[8, 4]\nC = A @ B\noutput C\n", opt=opt)
    est = comp.program.estimate
    assert est["instructions"]["min"] == est["instructions"]["max"]
    assert est["cycles"]["min"] == est["cycles"]["max"]
    res, _, _ = run_on_simulator(comp, 5)
    assert res.instructions == est["instructions"]["min"]
    assert res.cycles == est["cycles"]["min"]
    assert res.op_counts["SYNCB"] == interp_syncb(comp, res)


def interp_syncb(comp, res):
    inputs = ref.random_inputs(comp.func, np.random.default_rng(5))
    got = interp.run_program(comp.program, inputs)["kernels"]
    return sum(k.ops["SYNCB"] for k in got)


def test_matmul_cycles_are_the_loop_trip_count_plus_the_relu_cycle():
    comp = build(MM_RELU, opt=1)  # K = 8, loop kept
    res, _, _ = run_on_simulator(comp, 1)
    assert res.cycles in (8, 9)  # 8 iterations; +1 when some thread does not clamp


def test_unrolling_removes_the_loop_cycles():
    comp = build(MM_RELU, opt=2)  # the K loop is unrolled: no taken back edges at all
    res, _, _ = run_on_simulator(comp, 1)
    assert res.cycles in (2, 3)  # arrive at the barrier, resume, (+1 for a non-clamping thread)


def test_global_memory_persists_across_launches():
    comp = build((ROOT / "examples" / "mlp_gpu.tiny").read_text())
    assert len(comp.launch["kernels"]) == 2
    res, _, want = run_on_simulator(comp)  # layer 2 reads what layer 1 wrote
    assert_bit_exact(comp, res, want)


def test_unfused_o0_chain_matches_the_fused_o1_program():
    src = (ROOT / "examples" / "mm_bias_relu.tiny").read_text()
    o0, o1 = build(src, opt=0), build(src, opt=1)
    assert len(o0.launch["kernels"]) == 3 and len(o1.launch["kernels"]) == 1
    inputs = ref.random_inputs(o0.func, np.random.default_rng(9))
    a, b = run_launch(o0.launch, inputs), run_launch(o1.launch, inputs)
    for n0, n1 in zip(o0.launch["outputs"], o1.launch["outputs"]):
        np.testing.assert_array_equal(a.outputs[n0], b.outputs[n1])
    assert b.instructions < a.instructions  # fusion removes launches and memory traffic


def test_the_smallest_legal_register_budget_still_works():
    comp = build(MM_RELU, opt=0, regs=8)  # -O0 is the register-frugal code
    res, _, want = run_on_simulator(comp)
    assert_bit_exact(comp, res, want)


def test_fewer_than_eight_registers_is_rejected_at_compile_time():
    with pytest.raises(PassError, match="at least 8 registers"):
        build(MM_RELU, regs=7)


def test_extreme_int8_values_do_not_overflow_the_simulator():
    comp = build("input A : i8[4, 8]\ninput B : i8[8, 4]\nC = A @ B\noutput C\n")
    inputs = {"A": np.full((4, 8), -128), "B": np.full((8, 4), -128)}
    res = run_launch(comp.launch, inputs)
    assert (res.outputs["0"] == 8 * 128 * 128).all()


# --- the series' point: one program, two chips ------------------------------------------------


@pytest.mark.parametrize("path", [p for p in EXAMPLES if p.stem != "mlp_gpu"], ids=lambda p: p.stem)
def test_tiny_gpu_and_tiny_tpu_agree_on_the_same_program_and_inputs(path):
    gpu = build(path.read_text())
    tpu = compile_tpu(path.read_text(), path.name)
    inputs = ref.random_inputs(gpu.func, np.random.default_rng(21))
    on_gpu = run_launch(gpu.launch, inputs)
    on_tpu = run_ttpu(tpu.program, inputs)
    for g, t in zip(gpu.launch["outputs"], tpu.schedule.outputs):
        np.testing.assert_array_equal(on_gpu.outputs[g], on_tpu.outputs[t])
