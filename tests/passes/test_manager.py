import pytest

from tests.helpers import ir
from tinyc.driver import build_pipeline, compile_graph, pipeline_names
from tinyc.ir.printer import print_func
from tinyc.ir.types import TensorType
from tinyc.passes import REGISTRY, Fixpoint, PassContext, PassManager, PassVerifyError

SRC = "input A : i8[2, 3]\ninput B : i8[3, 2]\nC = relu(A @ B)\noutput C\n"


class DropRelu:
    """A deliberately wrong pass: pretends relu is the identity."""

    name = "drop-relu"

    def run(self, func, ctx):
        for op in func.ops:
            if op.kind == "relu":
                func.replace_uses(op.result, op.operands[0])


class WrongType:
    """A pass that corrupts a type: structurally invalid."""

    name = "wrong-type"

    def run(self, func, ctx):
        func.ops[0].result.type = TensorType("i32", (7, 7))


def func():
    return ir("""
        func @main(%A: i8[2,3], %B: i8[3,2]) -> (i32[2,2]) {
          %0 = matmul %A, %B : i32[2,2]
          %1 = relu %0 : i32[2,2]
          return %1
        }
        """)


def test_verify_each_catches_a_semantically_wrong_pass_and_shows_before_and_after():
    with pytest.raises(PassVerifyError) as exc:
        PassManager([DropRelu()], verify_each=True).run(func())
    err = exc.value
    assert err.pass_name == "drop-relu" and "differs from the oracle" in err.reason
    assert "return %1" in err.before and "return %0" in err.after


def test_without_verify_each_a_semantic_bug_goes_unnoticed():
    PassManager([DropRelu()]).run(func())  # opt-in: structure is fine, so nothing complains


def test_structural_verification_always_runs():
    with pytest.raises(PassVerifyError, match="operands give"):
        PassManager([WrongType()]).run(func())


def test_report_records_changes_timing_and_ir():
    f = func()
    report = PassManager([REGISTRY["fuse"](), REGISTRY["dce"]()]).run(f)
    assert [r.name for r in report.records] == ["fuse", "dce"]
    assert [r.changed for r in report.records] == [True, False]
    assert report.records[-1].ir == print_func(f) and "relu" in report.initial_ir
    assert all(r.seconds >= 0 for r in report.records)
    assert "fuse" in report.timing_table()


def test_dump_callback_sees_input_then_every_pass():
    seen = []
    PassManager([REGISTRY["fuse"](), REGISTRY["dce"]()], dump=lambda n, t: seen.append(n)).run(
        func()
    )
    assert seen == ["input", "fuse", "dce"]


def test_pipeline_names_per_target_and_level():
    assert pipeline_names(None, 0) == ["shape-infer", "range-check"]
    assert pipeline_names("tpu", 0)[-1] == "legalize-tpu"
    full = pipeline_names("gpu", 1)
    assert full[0] == "shape-infer" and full[-2:] == ["range-check", "legalize-gpu"]
    assert full.index("fuse") < full.index("range-check")


def test_disabling_passes_and_unknown_names():
    def flat(pipeline):
        return [q.name for p in pipeline for q in (p.passes if isinstance(p, Fixpoint) else [p])]

    names = flat(build_pipeline("gpu", 1, disabled=["fuse", "cse"]))
    assert "fuse" not in names and "cse" not in names and "dce" in names
    with pytest.raises(ValueError, match="unknown passes"):
        build_pipeline(None, 1, disabled=["nope"])


def test_o0_does_not_fuse_but_o1_does():
    f0, _ = compile_graph(SRC, "t.tiny", opt_level=0)
    f1, _ = compile_graph(SRC, "t.tiny", opt_level=1)
    assert [o.kind for o in f0.ops] == ["matmul", "relu"]
    assert [(o.kind, o.attrs) for o in f1.ops] == [("matmul", {"epilogue": "relu"})]


def test_fused_op_keeps_the_outermost_source_location():
    f, _ = compile_graph(SRC, "t.tiny")
    assert (f.ops[0].loc.line, f.ops[0].loc.col) == (3, 5)  # the relu( ... ) expression


def test_a_pipeline_run_twice_is_a_no_op_the_second_time():
    f, _ = compile_graph(SRC, "t.tiny", target="gpu")
    again = PassManager(build_pipeline("gpu", 1), verify_each=True).run(f, PassContext())
    assert not any(r.changed for r in again.records)


# --- fixpoint group ---------------------------------------------------------------------------

# Found by Hypothesis (shrunk): CSE merges v3 into the earlier v0, which leaves `v2 + v0` with its
# operands in non-canonical order. One canonicalize-then-cse sweep stops there.
CSE_BREAKS_CANONICAL_ORDER = """\
input x0 : i8[1]
input x1 : i8[2]
v0 = relu(x0)
v1 = relu(x1)
v2 = x0 + x0
v3 = relu(x0)
v4 = relu(x0)
v5 = v2 + v3
output v5
"""
SIMPLIFY = ("canonicalize", "const-fold", "cse", "dce")


def source_func():
    from tinyc.frontend import compile_to_graph

    return compile_to_graph(CSE_BREAKS_CANONICAL_ORDER, "t.tiny")


def test_one_sweep_of_the_simplifications_is_not_a_fixpoint():
    f = source_func()
    PassManager([REGISTRY[n]() for n in SIMPLIFY]).run(f)
    assert PassManager([REGISTRY["canonicalize"]()]).run(f).records[0].changed


def test_the_fixpoint_group_reaches_a_stable_program():
    f = source_func()
    group = Fixpoint([REGISTRY[n]() for n in SIMPLIFY])
    report = PassManager([group], verify_each=True).run(f)
    again = PassManager([group], verify_each=True).run(f)
    assert any(r.changed for r in report.records)
    assert not any(r.changed for r in again.records)
    assert len(report.records) > len(SIMPLIFY)  # more than one round, each sub-pass recorded


def test_a_fixpoint_group_that_never_settles_is_an_error():
    class Flip:
        name = "flip"

        def run(self, func, ctx):
            func.name = "a" if func.name != "a" else "b"

    with pytest.raises(RuntimeError, match="did not converge"):
        PassManager([Fixpoint([Flip()], max_iterations=3)]).run(func())
