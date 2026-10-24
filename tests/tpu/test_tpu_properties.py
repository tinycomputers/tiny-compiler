"""Random programs through the TPU backend: reference interpreter, and the real RTL."""

import numpy as np
from hypothesis import HealthCheck, event, given, settings
from hypothesis import strategies as st

from runtime.tpu_runner import run_ttpu
from tests.gen import programs, tpu_programs
from tests.tpu.helpers import build
from tinyc.ir import ref
from tinyc.ir.errors import PassError
from tinyc.tpu import interp


def _compile(source, opt):
    try:
        return build(source, opt=opt, verify_each=True)
    except PassError:  # not tiny-tpu legal (i32 matmul operand) or overflow not provable
        event("rejected by legalization or range-check")
        return None


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.one_of(tpu_programs(), programs()), st.sampled_from([0, 1]), st.integers(0, 2**32 - 1))
def test_reference_interpreter_matches_oracle_on_random_programs(source, opt, seed):
    comp = _compile(source, opt)
    if comp is None:
        return
    inputs = ref.random_inputs(comp.func, np.random.default_rng(seed))
    got = interp.run_program(comp.program, inputs)
    assert got["passes"] == comp.program["estimate"]["passes"]
    for name, want in zip(comp.schedule.outputs, ref.evaluate(comp.func, inputs)):
        np.testing.assert_array_equal(got["outputs"][name], want)


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(tpu_programs(), st.sampled_from([0, 1]), st.integers(0, 2**32 - 1))
def test_real_rtl_matches_oracle_and_the_cost_model_on_random_programs(source, opt, seed):
    comp = _compile(source, opt)
    if comp is None:
        return
    est = comp.program["estimate"]
    inputs = ref.random_inputs(comp.func, np.random.default_rng(seed))
    got = run_ttpu(comp.program, inputs)
    for name, want in zip(comp.schedule.outputs, ref.evaluate(comp.func, inputs)):
        np.testing.assert_array_equal(got.outputs[name], want)
    assert got.total_cycles == est["rtl_cycles"] and got.host_ops == est["host_ops"]
