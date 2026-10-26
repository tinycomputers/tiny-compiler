import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tinyc.frontend import compile_to_graph
from tinyc.ir import interp, ref
from tinyc.ir.errors import InputError, RangeError, VerifyError
from tinyc.ir.graph import Func, Op, Value
from tinyc.ir.parser import parse_func
from tinyc.ir.printer import print_func
from tinyc.ir.types import TensorType
from tinyc.ir.verifier import verify

MM_RELU = "input A : i8[2, 3]\ninput B : i8[3, 2]\nC = relu(A @ B)\noutput C\n"


def func(text=MM_RELU):
    return compile_to_graph(text, "t.tiny")


# --- printer / parser round trip ------------------------------------------------------------


@pytest.mark.parametrize("locs", [False, True])
def test_print_parse_print_is_stable(locs):
    f = func("input A : i8[2, 2]\nconst W : i8[2, 2] = [[1, -2], [3, 4]]\n"
             "input b : i32[2]\nY = relu(A @ W + b)\noutput Y, W\n")  # fmt: skip
    text = print_func(f, locs=locs)
    assert print_func(parse_func(text), locs=locs) == text


def test_ir_text_matches_the_plan_format():
    assert print_func(func(), locs=True) == (
        "func @main(%A: i8[2,3], %B: i8[3,2]) -> (i32[2,2]) {\n"
        '  %0 = matmul %A, %B : i32[2,2]  loc("t.tiny":3:10)\n'
        '  %1 = relu %0 : i32[2,2]        loc("t.tiny":3:5)\n'
        "  return %1\n"
        "}\n"
    )


def test_ir_parser_accepts_comments_and_free_layout():
    f = parse_func("// c\nfunc @main(%A: i8[2]) -> (i8[2]) { %0 = relu %A : i8[2]\n return %0 }")
    assert [o.kind for o in f.ops] == ["relu"]


@pytest.mark.parametrize(
    "text, fragment",
    [
        (
            "func @m(%A: i8[2]) -> (i8[2]) {\n  %0 = relu %Z : i8[2]\n  return %0\n}",
            "undefined value",
        ),
        ("func @m(%A: i8[2]) -> (i8[2]) {\n  %0 = frob %A : i8[2]\n  return %0\n}", "unknown op"),
        ("func @m(%A: i8[2]) -> (i8[3]) {\n  return %A\n}", "declared result types"),
        (
            "func @m(%A: i8[2]) -> (i8[2]) {\n  %A = relu %A : i8[2]\n  return %A\n}",
            "already defined",
        ),
        ("func @m() -> (i8[2]) {\n  %0 = const [[1], [2]] : i8[2]\n  return %0\n}", "shape"),
        ("func @m() -> (i8[2]) {\n  %0 = const [1, [2]] : i8[2]\n  return %0\n}", "ragged"),
    ],
)
def test_ir_parser_errors(text, fragment):
    with pytest.raises(Exception, match=fragment):
        parse_func(text)


# --- verifier -------------------------------------------------------------------------------


def test_verifier_accepts_frontend_output():
    verify(func())


def test_verifier_reports_every_problem():
    f = func()
    f.ops[0].result.type = TensorType("i32", (9, 9))  # wrong inferred type
    f.ops[1].operands[0] = Value("0", f.ops[0].result.type)  # not the defined object
    f.outputs = []
    with pytest.raises(VerifyError) as exc:
        verify(f)
    text = str(exc.value)
    assert "operands give" in text and "not defined before use" in text and "no outputs" in text


def test_verifier_catches_use_before_def_and_duplicates():
    a = Value("A", TensorType("i8", (2,)))
    r = Value("0", TensorType("i8", (2,)))
    f = Func("m", [a, a], [Op("relu", [r], r)], [r])
    with pytest.raises(VerifyError) as exc:
        verify(f)
    assert "defined twice" in str(exc.value) and "not defined before use" in str(exc.value)


def test_verifier_checks_const_data():
    f = func("const W : i8[1, 2] = [[1, 2]]\noutput W\n")
    f.ops[0].data = np.array([[1, 500]], dtype=np.int64)
    with pytest.raises(VerifyError, match="does not fit"):
        verify(f)
    f.ops[0].data = np.array([1, 2], dtype=np.int64)
    with pytest.raises(VerifyError, match="shape"):
        verify(f)


def test_verifier_rejects_illegal_types():
    f = func()
    f.args[0].type = TensorType("i8", (2, 2, 2))
    with pytest.raises(VerifyError, match="rank"):
        verify(f)


# --- semantics: hand-checked values ---------------------------------------------------------


def test_matmul_relu_by_hand():
    f = func()
    a = [[1, -2, 3], [0, 4, -1]]
    b = [[1, 0], [2, -1], [-1, 5]]
    expect = np.maximum(np.array(a) @ np.array(b), 0)  # [[-6, 17], [9, -9]] -> relu
    for run in (ref.evaluate, interp.run):
        (out,) = run(f, {"A": a, "B": b})
        assert out.tolist() == [[0, 17], [9, 0]] == expect.tolist()


@pytest.mark.parametrize("order", ["M + b", "b + M"])
def test_add_row_broadcast_in_either_order(order):
    f = func(f"input M : i8[2, 3]\ninput b : i32[3]\nY = {order}\noutput Y\n")
    inputs = {"M": [[1, 2, 3], [4, 5, 6]], "b": [10, 20, 30]}
    for run in (ref.evaluate, interp.run):
        assert run(f, inputs)[0].tolist() == [[11, 22, 33], [14, 25, 36]]


def test_rank1_add_and_relu():
    f = func("input a : i8[3]\ninput b : i8[3]\nY = relu(a + b)\noutput Y\n")
    for run in (ref.evaluate, interp.run):
        assert run(f, {"a": [-5, 2, 3], "b": [1, 1, -9]})[0].tolist() == [0, 3, 0]


def test_i8_operands_widen_so_extremes_do_not_wrap():
    f = func("input a : i8[1, 2]\ninput b : i8[2, 1]\nC = a @ b\noutput C\n")
    for run in (ref.evaluate, interp.run):
        assert run(f, {"a": [[-128, -128]], "b": [[-128], [-128]]})[0].tolist() == [[32768]]


def test_multiple_outputs_come_back_in_order():
    f = func("input A : i8[2]\nR = relu(A)\noutput R, A\n")
    r, a = interp.run(f, {"A": [-1, 2]})
    assert r.tolist() == [0, 2] and a.tolist() == [-1, 2]


def test_i32_overflow_raises_in_both_evaluators():
    f = func("input a : i32[1, 2]\ninput b : i32[2, 1]\nC = a @ b\noutput C\n")
    big = {"a": [[2**20, 2**20]], "b": [[2**20], [2**20]]}  # 2 * 2**40
    for run in (ref.evaluate, interp.run):
        with pytest.raises(RangeError):
            run(f, big)


@pytest.mark.parametrize(
    "inputs, fragment",
    [
        ({"A": [[0, 0, 0]] * 2}, "missing input"),
        ({"A": [[0, 0, 0]] * 2, "B": [[0, 0]] * 3, "Z": 1}, "unexpected"),
        ({"A": [[0, 0]], "B": [[0, 0]] * 3}, "shape"),
        ({"A": [[0, 0, 200]] * 2, "B": [[0, 0]] * 3}, "outside i8"),
    ],
)
def test_input_validation(inputs, fragment):
    for run in (ref.evaluate, interp.run):
        with pytest.raises(InputError, match=fragment):
            run(func(), inputs)


# --- the two evaluators are independent implementations: cross-check them -------------------


def test_random_inputs_are_seeded():
    f = func()
    a = ref.random_inputs(f, np.random.default_rng(42))
    b = ref.random_inputs(f, np.random.default_rng(42))
    assert all((a[k] == b[k]).all() for k in a)


@settings(max_examples=60, deadline=None)
@given(
    m=st.integers(1, 6), k=st.integers(1, 6), n=st.integers(1, 6),
    seed=st.integers(0, 2**32 - 1), bias=st.booleans(), second=st.booleans(),
)  # fmt: skip
def test_interp_equals_oracle_on_random_programs(m, k, n, seed, bias, second):
    lines = [f"input A : i8[{m}, {k}]", f"input B : i8[{k}, {n}]", f"input c : i32[{n}]",
             f"input D : i8[{n}, {m}]"]  # fmt: skip
    lines.append("H = relu(A @ B + c)" if bias else "H = A @ B")
    lines.append("Y = H @ D" if second else "Y = H")
    lines.append("output Y")
    f = func("\n".join(lines) + "\n")
    inputs = ref.random_inputs(f, np.random.default_rng(seed))
    (want,), (got,) = ref.evaluate(f, inputs), interp.run(f, inputs)
    np.testing.assert_array_equal(got, want)
