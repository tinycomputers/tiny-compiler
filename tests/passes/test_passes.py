"""Directed tests: IR text in, expected IR text out, one group per pass."""

import pytest

from tests.helpers import expected, ir, run_passes
from tinyc.ir.errors import PassError
from tinyc.passes import PassContext, PassManager
from tinyc.passes.range_check import RangeCheck

# --- canonicalize -----------------------------------------------------------------------------


def test_canonicalize_puts_the_matrix_first_in_a_broadcast_add():
    out = run_passes(
        """
        func @main(%M: i8[2,3], %b: i8[3]) -> (i32[2,3]) {
          %0 = add %b, %M : i32[2,3]
          return %0
        }
        """,
        "canonicalize",
    )
    assert "add %M, %b" in out


def test_canonicalize_orders_same_shape_adds_by_definition_order():
    out = run_passes(
        """
        func @main(%x: i8[2], %y: i8[2]) -> (i32[2]) {
          %0 = add %y, %x : i32[2]
          return %0
        }
        """,
        "canonicalize",
    )
    assert "add %x, %y" in out


def test_canonicalize_collapses_double_relu():
    out = run_passes(
        """
        func @main(%A: i8[2]) -> (i8[2]) {
          %0 = relu %A : i8[2]
          %1 = relu %0 : i8[2]
          return %1
        }
        """,
        "canonicalize",
        "dce",
    )
    assert out == expected("""
        func @main(%A: i8[2]) -> (i8[2]) {
          %0 = relu %A : i8[2]
          return %0
        }
        """)


# --- const-fold -------------------------------------------------------------------------------


def test_const_fold_evaluates_constant_chains_and_dce_cleans_up():
    out = run_passes(
        """
        func @main(%A: i8[2,2]) -> (i32[2,2]) {
          %0 = const [[1, 2], [3, 4]] : i8[2,2]
          %1 = const [[1, 0], [0, 1]] : i8[2,2]
          %2 = matmul %0, %1 : i32[2,2]
          %3 = add %2, %2 : i32[2,2]
          return %3
        }
        """,
        "const-fold",
        "dce",
    )
    assert out == expected("""
        func @main(%A: i8[2,2]) -> (i32[2,2]) {
          %3 = const [[2, 4], [6, 8]] : i32[2,2]
          return %3
        }
        """)


def test_const_fold_folds_relu_of_a_constant_and_keeps_its_dtype():
    out = run_passes(
        """
        func @main() -> (i8[3]) {
          %0 = const [-5, 0, 7] : i8[3]
          %1 = relu %0 : i8[3]
          return %1
        }
        """,
        "const-fold",
    )
    assert "%1 = const [0, 0, 7] : i8[3]" in out


def test_const_fold_leaves_ops_with_a_non_constant_operand():
    text = """
        func @main(%A: i8[2]) -> (i32[2]) {
          %0 = const [1, 2] : i8[2]
          %1 = add %A, %0 : i32[2]
          return %1
        }
        """
    assert run_passes(text, "const-fold") == expected(text)


def test_const_fold_reports_overflowing_constant_expressions():
    big = ", ".join(["2000000000"] * 2)
    f = ir(f"""
        func @main() -> (i32[2]) {{
          %0 = const [{big}] : i32[2]
          %1 = add %0, %0 : i32[2]
          return %1
        }}
        """)
    from tinyc.passes import REGISTRY

    with pytest.raises(PassError, match="overflows i32"):
        REGISTRY["const-fold"]().run(f, PassContext())


# --- cse --------------------------------------------------------------------------------------


def test_cse_merges_identical_ops():
    out = run_passes(
        """
        func @main(%A: i8[2,2], %B: i8[2,2]) -> (i32[2,2]) {
          %0 = matmul %A, %B : i32[2,2]
          %1 = matmul %A, %B : i32[2,2]
          %2 = add %0, %1 : i32[2,2]
          return %2
        }
        """,
        "cse",
    )
    assert out == expected("""
        func @main(%A: i8[2,2], %B: i8[2,2]) -> (i32[2,2]) {
          %0 = matmul %A, %B : i32[2,2]
          %2 = add %0, %0 : i32[2,2]
          return %2
        }
        """)


def test_cse_merges_equal_constants_but_not_different_dtypes():
    out = run_passes(
        """
        func @main() -> (i8[2], i8[2], i32[2]) {
          %0 = const [1, 2] : i8[2]
          %1 = const [1, 2] : i8[2]
          %2 = const [1, 2] : i32[2]
          return %0, %1, %2
        }
        """,
        "cse",
    )
    assert out.count("const") == 2 and "return %0, %0, %2" in out


def test_canonicalize_then_cse_sees_through_commutativity():
    out = run_passes(
        """
        func @main(%x: i8[2], %y: i8[2]) -> (i32[2]) {
          %0 = add %x, %y : i32[2]
          %1 = add %y, %x : i32[2]
          %2 = add %0, %1 : i32[2]
          return %2
        }
        """,
        "canonicalize",
        "cse",
    )
    assert "%2 = add %0, %0" in out and "%1" not in out


# --- dce --------------------------------------------------------------------------------------


def test_dce_removes_dead_ops_and_keeps_unused_inputs():
    out = run_passes(
        """
        func @main(%A: i8[2], %B: i8[2]) -> (i8[2]) {
          %0 = relu %A : i8[2]
          %1 = add %A, %B : i32[2]
          %2 = relu %1 : i32[2]
          return %0
        }
        """,
        "dce",
    )
    assert out == expected("""
        func @main(%A: i8[2], %B: i8[2]) -> (i8[2]) {
          %0 = relu %A : i8[2]
          return %0
        }
        """)


# --- fuse -------------------------------------------------------------------------------------

MM_BIAS_RELU = """
    func @main(%A: i8[2,3], %B: i8[3,2], %b: i32[2]) -> (i32[2,2]) {
      %0 = matmul %A, %B : i32[2,2]
      %1 = add %0, %b : i32[2,2]
      %2 = relu %1 : i32[2,2]
      return %2
    }
    """


def test_fuse_matmul_bias_relu():
    assert run_passes(MM_BIAS_RELU, "fuse") == expected("""
        func @main(%A: i8[2,3], %B: i8[3,2], %b: i32[2]) -> (i32[2,2]) {
          %2 = matmul{epilogue=bias,relu} %A, %B, %b : i32[2,2]
          return %2
        }
        """)


def test_fuse_matmul_relu_and_matmul_bias_alone():
    out = run_passes(
        """
        func @main(%A: i8[2,3], %B: i8[3,2]) -> (i32[2,2]) {
          %0 = matmul %A, %B : i32[2,2]
          %1 = relu %0 : i32[2,2]
          return %1
        }
        """,
        "fuse",
    )
    assert "%1 = matmul{epilogue=relu} %A, %B : i32[2,2]" in out
    out = run_passes(
        """
        func @main(%A: i8[2,3], %B: i8[3,2], %b: i8[2]) -> (i32[2,2]) {
          %0 = matmul %A, %B : i32[2,2]
          %1 = add %b, %0 : i32[2,2]
          return %1
        }
        """,
        "fuse",
    )
    assert "%1 = matmul{epilogue=bias} %A, %B, %b : i32[2,2]" in out  # either operand order


def test_fuse_moves_the_op_after_a_late_defined_bias():
    out = run_passes(
        """
        func @main(%A: i8[2,3], %B: i8[3,2]) -> (i32[2,2]) {
          %0 = matmul %A, %B : i32[2,2]
          %1 = const [1, 2] : i32[2]
          %2 = add %0, %1 : i32[2,2]
          return %2
        }
        """,
        "fuse",
    )
    assert out == expected("""
        func @main(%A: i8[2,3], %B: i8[3,2]) -> (i32[2,2]) {
          %1 = const [1, 2] : i32[2]
          %2 = matmul{epilogue=bias} %A, %B, %1 : i32[2,2]
          return %2
        }
        """)


@pytest.mark.parametrize(
    "body",
    [
        # the matmul result has two users
        "%0 = matmul %A, %B : i32[2,2]\n%1 = relu %0 : i32[2,2]\n%2 = add %0, %1 : i32[2,2]\nreturn %2",
        # the matmul result is also an output
        "%0 = matmul %A, %B : i32[2,2]\n%1 = relu %0 : i32[2,2]\nreturn %0, %1",
        # add with a full matrix is not a bias
        "%0 = matmul %A, %B : i32[2,2]\n%1 = add %0, %0 : i32[2,2]\nreturn %1",
    ],
)
def test_fuse_leaves_unfusable_patterns_alone(body):
    rets = "i32[2,2], i32[2,2]" if "return %0, %1" in body else "i32[2,2]"
    text = f"func @main(%A: i8[2,3], %B: i8[3,2]) -> ({rets}) {{\n{body}\n}}"
    assert run_passes(text, "fuse") == run_passes(text)  # unchanged


def test_fuse_is_idempotent():
    once = run_passes(MM_BIAS_RELU, "fuse")
    assert run_passes(once, "fuse") == once


# --- range-check ------------------------------------------------------------------------------


def _mm(k, a="i8", b="i8"):
    return f"""
    func @main(%A: {a}[1,{k}], %B: {b}[{k},1]) -> (i32[1,1]) {{
      %0 = matmul %A, %B : i32[1,1]
      return %0
    }}
    """


def test_range_check_computes_exact_interval_bounds():
    ctx = PassContext()
    f = ir(_mm(4))
    PassManager([RangeCheck()]).run(f, ctx)
    # i8 x i8 products lie in [-128*127, 128*128] = [-16256, 16384]; four of them:
    assert ctx.analysis["ranges"]["0"] == (-65024, 65536)
    assert ctx.analysis["ranges"]["A"] == (-128, 127)


def test_range_check_boundary_is_exact():
    PassManager([RangeCheck()]).run(ir(_mm(131071)), PassContext())  # 16384 * 131071 < 2**31
    with pytest.raises(PassError, match="does not fit in i32") as exc:
        PassManager([RangeCheck()]).run(ir(_mm(131072)), PassContext())  # exactly 2**31
    assert "matmul" in str(exc.value)


def test_range_check_assumes_a_range_for_i32_inputs_and_says_so():
    with pytest.raises(PassError, match="i32 inputs are assumed"):
        PassManager([RangeCheck()]).run(ir(_mm(4, "i32", "i32")), PassContext())
    wide = PassContext(i32_input_range=(-1000, 1000))
    PassManager([RangeCheck()]).run(ir(_mm(4, "i32", "i32")), wide)  # 4 * 1000 * 1000 fits


def test_range_check_tracks_relu_bias_and_constants_exactly():
    ctx = PassContext()
    f = ir("""
        func @main(%A: i8[1,2]) -> (i32[1,2]) {
          %0 = const [[100, -3], [2, 5]] : i8[2,2]
          %1 = matmul{epilogue=relu} %A, %0 : i32[1,2]
          return %1
        }
        """)
    PassManager([RangeCheck()]).run(f, ctx)
    ranges = ctx.analysis["ranges"]
    assert ranges["0"] == (-3, 100)  # a constant is bounded by its own min and max
    # products of [-128,127] x [-3,100] lie in [-12800, 12700]; two of them, then relu:
    assert ranges["1"] == (0, 25400)


def test_range_check_rejects_overflowing_add_chain():
    f = ir("""
        func @main(%a: i32[1], %b: i32[1]) -> (i32[1]) {
          %0 = add %a, %b : i32[1]
          return %0
        }
        """)
    PassManager([RangeCheck()]).run(f, PassContext())  # 2 * 2**20 fits
    with pytest.raises(PassError, match="add"):
        PassManager([RangeCheck()]).run(f, PassContext(i32_input_range=(-(2**31), 2**31 - 1)))


# --- legalization -----------------------------------------------------------------------------


def test_tpu_rejects_i32_operands_with_a_clear_message():
    f = ir("""
        func @main(%A: i8[2,2], %B: i8[2,2], %D: i8[2,2]) -> (i32[2,2]) {
          %0 = matmul %A, %B : i32[2,2]
          %1 = matmul %0, %D : i32[2,2]
          return %1
        }
        """)
    from tinyc.passes import REGISTRY

    with pytest.raises(PassError, match=r"tiny-tpu needs i8 matmul operands, but %0 is i32\[2,2\]"):
        REGISTRY["legalize-tpu"]().run(f, PassContext())
    REGISTRY["legalize-gpu"]().run(f, PassContext())  # the GPU runs chained matmuls


def test_tpu_rejects_computed_i8_operands_but_accepts_constants():
    from tinyc.passes import REGISTRY

    computed = ir("""
        func @main(%A: i8[2,2], %B: i8[2,2]) -> (i32[2,2]) {
          %0 = relu %A : i8[2,2]
          %1 = matmul %0, %B : i32[2,2]
          return %1
        }
        """)
    with pytest.raises(PassError, match="computed by 'relu'"):
        REGISTRY["legalize-tpu"]().run(computed, PassContext())
    const = ir("""
        func @main(%A: i8[2,2]) -> (i32[2,2]) {
          %0 = const [[1, 2], [3, 4]] : i8[2,2]
          %1 = matmul %A, %0 : i32[2,2]
          return %1
        }
        """)
    REGISTRY["legalize-tpu"]().run(const, PassContext())


def test_gpu_limits_on_threads_and_memory():
    from tinyc.passes import REGISTRY

    REGISTRY["legalize-gpu"]().run(ir(_mm(4)), PassContext())
    square = ir("""
        func @main(%A: i8[3,2], %B: i8[2,3]) -> (i32[3,3]) {
          %0 = matmul %A, %B : i32[3,3]
          return %0
        }
        """)
    with pytest.raises(PassError, match="9 threads"):
        REGISTRY["legalize-gpu"]().run(square, PassContext(gpu_max_threads=8))
    with pytest.raises(PassError, match="words of tiny-gpu memory"):
        REGISTRY["legalize-gpu"]().run(square, PassContext(gpu_mem_words=20))


# --- shape-infer ------------------------------------------------------------------------------


def test_shape_infer_repairs_stale_types():
    f = ir("""
        func @main(%A: i8[2,3], %B: i8[3,2]) -> (i32[2,2]) {
          %0 = matmul %A, %B : i32[2,2]
          return %0
        }
        """)
    from tinyc.ir.types import TensorType
    from tinyc.passes import REGISTRY

    f.ops[0].result.type = TensorType("i32", (9, 9))
    REGISTRY["shape-infer"]().run(f, PassContext())
    assert str(f.ops[0].result.type) == "i32[2,2]"
