"""Random programs through the GPU backend: the interpreter, and the real simulator."""

import numpy as np
from hypothesis import HealthCheck, event, given, settings
from hypothesis import strategies as st

from tests.gen import programs, tpu_programs
from tests.gpu.helpers import build, run_on_simulator
from tinyc.gpu import interp
from tinyc.ir import ref
from tinyc.ir.errors import PassError

SOURCES = st.one_of(programs(), tpu_programs())


def _compile(source, opt):
    try:
        return build(
            source, opt=opt, verify_each=True
        )  # kernel interpreter AND simulator each stage
    except PassError:
        event("rejected by range-check or legalization")
        return None


@settings(max_examples=120, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(SOURCES, st.sampled_from([0, 1]), st.integers(0, 2**32 - 1))
def test_kernel_interpreter_matches_oracle_on_random_programs(source, opt, seed):
    comp = _compile(source, opt)
    if comp is None:
        return
    inputs = ref.random_inputs(comp.func, np.random.default_rng(seed))
    got = interp.run_program(comp.program, inputs)["outputs"]
    for name, want in zip(comp.launch["outputs"], ref.evaluate(comp.func, inputs)):
        np.testing.assert_array_equal(got[name], want)


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(SOURCES, st.sampled_from([0, 1]), st.integers(0, 2**32 - 1))
def test_simulator_matches_oracle_and_interpreter_counts_on_random_programs(source, opt, seed):
    comp = _compile(source, opt)
    if comp is None:
        return
    res, inputs, want = run_on_simulator(comp, seed)
    assert res.reserved_intact
    for name, w in zip(comp.launch["outputs"], want):
        np.testing.assert_array_equal(res.outputs[name], w)
    for (_, sim), counts in zip(res.kernels, interp.run_program(comp.program, inputs)["kernels"]):
        assert dict(sim.op_counts) == dict(counts.ops) and sim.cycles == counts.cycles
