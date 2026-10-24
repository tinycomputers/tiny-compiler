"""The whole pipeline over every example, and over random programs."""

from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, event, given, settings
from hypothesis import strategies as st

from tests.gen import programs
from tinyc.driver import build_pipeline, compile_graph
from tinyc.frontend import compile_to_graph
from tinyc.ir import ref
from tinyc.ir.errors import PassError, RangeError
from tinyc.ir.parser import parse_func
from tinyc.ir.printer import print_func
from tinyc.passes import PassContext, PassManager

ROOT = Path(__file__).resolve().parent.parent.parent
EXAMPLES = sorted((ROOT / "examples").glob("*.tiny"))
TPU_ILLEGAL = {"mlp_gpu"}  # its second matmul has an i32 operand


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
@pytest.mark.parametrize("opt", [0, 1])
@pytest.mark.parametrize("target", [None, "gpu", "tpu"])
def test_examples_through_every_pipeline_with_verify_each(path, opt, target):
    if target == "tpu" and path.stem in TPU_ILLEGAL:
        with pytest.raises(PassError, match="tiny-tpu needs i8"):
            compile_graph(path.read_text(), path.name, target=target, opt_level=opt)
        return
    func, _ = compile_graph(
        path.read_text(), path.name, target=target, opt_level=opt, verify_each=True
    )
    # the optimized program still computes what the source computes
    source = compile_to_graph(path.read_text(), path.name)
    rng = np.random.default_rng(7)
    for _ in range(5):
        inputs = ref.random_inputs(source, rng)
        for got, want in zip(ref.evaluate(func, inputs), ref.evaluate(source, inputs)):
            np.testing.assert_array_equal(got, want)
    # and the optimized IR survives a print/parse round trip, attributes included
    text = print_func(func, locs=True)
    assert print_func(parse_func(text), locs=True) == text


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_optimized_ir_golden(path, golden):
    func, _ = compile_graph(path.read_text(), path.name)
    golden(ROOT / "tests" / "golden" / "opt" / f"{path.stem}.ir", print_func(func, locs=True))


def test_dense_layer_fuses_completely_and_mlp_fuses_the_first_layer():
    func, _ = compile_graph((ROOT / "examples" / "mm_bias_relu.tiny").read_text(), "x.tiny")
    assert [(o.kind, o.attrs.get("epilogue")) for o in func.ops] == [("matmul", "bias,relu")]
    func, _ = compile_graph((ROOT / "examples" / "mlp_gpu.tiny").read_text(), "x.tiny")
    assert [(o.kind, o.attrs.get("epilogue")) for o in func.ops] == [
        ("matmul", "bias,relu"),
        ("matmul", None),
    ]


# --- properties over random programs ----------------------------------------------------------


def _try(source: str, **kw):
    try:
        return compile_graph(source, "gen.tiny", **kw)
    except PassError:  # range_check proved it could overflow: a legitimate rejection
        event("rejected by range-check")
        return None


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(programs())
def test_pipeline_preserves_semantics_on_random_programs(source):
    # verify_each re-runs the program after *every* pass and compares with the source's oracle
    _try(source, verify_each=True)


@settings(max_examples=200, deadline=None)
@given(programs())
def test_pipeline_is_idempotent(source):
    got = _try(source)
    if got is None:
        return
    func, _ = got
    again = PassManager(build_pipeline(None, 1), verify_each=True).run(func, PassContext())
    assert not any(r.changed for r in again.records), [r.name for r in again.records if r.changed]


@settings(max_examples=200, deadline=None)
@given(programs(), st.integers(0, 2**32 - 1))
def test_range_check_is_sound_on_extreme_inputs(source, seed):
    """If range-check accepts a program, no input in the assumed ranges can overflow, and every
    value stays inside its proven interval. Inputs are drawn from the interval corners."""
    ctx = PassContext()
    got = _try(source, opt_level=0, ctx=ctx)
    if got is None:
        return
    func, _ = got
    rng = np.random.default_rng(seed)
    inputs = {}
    for a in func.args:
        lo, hi = (-128, 127) if a.type.dtype == "i8" else ctx.i32_input_range
        inputs[a.name] = rng.choice([lo, hi], size=a.type.shape).astype(np.int64)
    env = dict(inputs)
    for op in func.ops:
        try:
            env[op.result.name] = ref.eval_op(
                op.kind, [env[v.name] for v in op.operands], op.attrs, op.data, op.result.name
            )
        except RangeError as e:  # pragma: no cover - would be a soundness bug
            pytest.fail(f"accepted program overflowed: {e}\n{source}")
    for name, (lo, hi) in ctx.analysis["ranges"].items():
        arr = env[name]
        assert lo <= arr.min() and arr.max() <= hi, f"%{name} escaped its interval\n{source}"
