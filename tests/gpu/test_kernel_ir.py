import re

import pytest

from tests.gpu.helpers import EXAMPLES, ROOT, build
from tinyc.gpu.kernel_ir import (
    Assign,
    For,
    If,
    Kernel,
    KernelError,
    Load,
    Special,
    Store,
    VReg,
    flatten,
    print_kernel,
    verify_kernel,
)

v = [VReg(i) for i in range(8)]


def kernel(body, grid=(1, 4)):
    return Kernel("k", grid, body, "0", "relu")


def kernels(source, opt=1):
    return build(source, opt=opt).program.kernels


# --- lowering ---------------------------------------------------------------------------------


def test_each_graph_op_is_one_kernel_and_fusion_removes_launches():
    src = "input A : i8[4, 8]\ninput B : i8[8, 4]\nC = relu(A @ B)\noutput C\n"
    assert [k.kind for k in kernels(src, opt=0)] == ["matmul", "relu"]
    assert [k.kind for k in kernels(src, opt=1)] == ["matmul"]  # fused: one launch


def test_grid_is_rows_by_columns_and_rank1_uses_one_block():
    (k,) = kernels("input A : i8[4, 8]\ninput B : i8[8, 6]\nC = A @ B\noutput C\n")
    assert k.grid == (4, 6) and k.threads == 24
    (k,) = kernels("input a : i8[5]\ninput b : i8[5]\nC = a + b\noutput C\n")
    assert k.grid == (1, 5)


def test_broadcast_add_indexes_the_bias_by_column_and_same_shape_by_flat_id():
    text = print_kernel(kernels("input M : i8[3, 4]\ninput b : i8[4]\nY = M + b\noutput Y\n")[0])
    assert "add thread," in text and "add tid," in text


def test_chained_matmuls_are_two_launches_over_shared_memory():
    ks = kernels((ROOT / "examples" / "mlp_gpu.tiny").read_text())
    assert [(k.kind, k.grid) for k in ks] == [("matmul", (4, 6)), ("matmul", (4, 3))]


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_kernel_ir_golden(path, golden):
    prog = build(path.read_text()).program
    text = "\n".join(print_kernel(k) for k in prog.kernels)
    golden(ROOT / "tests" / "golden" / "kernel" / f"{path.stem}.kernel", text)


def test_matmul_kernel_has_the_naive_o0_shape():
    text = print_kernel(kernels("input A : i8[4, 8]\ninput B : i8[8, 4]\nC = A @ B\noutput C\n")[0])
    assert "for v1 in 0..8 {" in text
    assert (
        text.count("= ld [") == 2 and text.count("= mul ") == 3
    )  # A and B recomputed every iteration


# --- flattening idioms --------------------------------------------------------------------------


def test_for_and_if_flatten_to_tiny_gpus_idioms():
    k = kernel(
        [
            Assign("set", v[0], (0,)),
            For(v[1], 0, 3, [Assign("add", v[0], (v[0], v[1]))]),
            If(v[0], 0, [Assign("set", v[0], (0,))]),
            Store(Special("tid"), v[0]),
        ]
    )
    ops = [(i.op, i.label) for i in flatten(k)]
    assert ops == [
        ("SET", None), ("SET", None), ("LABEL", "L0"), ("ADD", None), ("ADD", None), ("CMP", None),
        ("BRLT", "L0"), ("CMP", None), ("BRGT", "L1"), ("SET", None), ("LABEL", "L1"), ("ST", None),
    ]  # fmt: skip


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_labels_never_start_with_r_because_the_assembler_would_parse_them_as_registers(path):
    for k in build(path.read_text()).program.kernels:
        for inst in flatten(k):
            if inst.label:
                assert re.match(r"L\d+$", inst.label), inst.label


# --- the verifier -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body, fragment",
    [
        ([Assign("add", v[0], (v[1], 1)), Store(Special("tid"), v[0])], "v1 used before"),
        ([Store(Special("tid"), v[0])], "v0 used before"),
        ([For(v[0], 3, 3, [Assign("set", v[1], (0,))]), Store(0, v[1])], "stop > start"),
        ([Assign("set", v[0], (2**31,)), Store(0, v[0])], "does not fit in i32"),
        ([Assign("set", v[0], (1, 2)), Store(0, v[0])], "'set' with 2 operands"),
        ([Assign("neg", v[0], (1,)), Store(0, v[0])], "'neg'"),
        ([Assign("set", v[0], (Special("warp"),)), Store(0, v[0])], "unknown special"),
        (
            [Assign("set", v[0], (0,)), If(v[0], 0, [Assign("set", v[0], (1,))])],
            "must not end with an `if`",
        ),
        (  # assigned only inside an `if`: not definitely assigned afterwards
            [Assign("set", v[0], (0,)), If(v[0], 0, [Assign("set", v[1], (1,))]), Store(0, v[1])],
            "v1 used before",
        ),
    ],
)
def test_kernel_verifier_rejects(body, fragment):
    with pytest.raises(KernelError, match=fragment):
        verify_kernel(kernel(body))


def test_kernel_verifier_rejects_oversized_grids():
    with pytest.raises(KernelError, match="1024-thread"):
        verify_kernel(kernel([Store(0, 0)], grid=(33, 32)))
    with pytest.raises(KernelError, match="positive"):
        verify_kernel(kernel([Store(0, 0)], grid=(0, 4)))


def test_values_assigned_in_a_for_body_are_definite_after_it_because_it_runs_at_least_once():
    verify_kernel(kernel([For(v[0], 0, 2, [Assign("set", v[1], (5,))]), Store(0, v[1])]))


def test_load_and_store_operands_are_checked():
    verify_kernel(kernel([Load(v[0], 3), Store(Special("tid"), v[0])]))
    with pytest.raises(KernelError, match="used before"):
        verify_kernel(kernel([Load(v[0], v[5]), Store(0, v[0])]))
