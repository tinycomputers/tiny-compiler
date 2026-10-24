import pytest

from tests.helpers import expected
from tests.tpu.helpers import ZERO_K_TILE, build, check_against_oracle
from tinyc.tpu.cost import estimate
from tinyc.tpu.tile_ir import Acc, Epilogue, Pass, print_schedule


def test_a_zero_constant_k_tile_loses_its_pass_and_the_epilogue_moves_earlier():
    comp = build(ZERO_K_TILE)
    assert print_schedule(comp.schedule) == expected("""
        tpu.schedule @main(%X: i8[4,8]) -> (%2) tile=4x4 {
          host.const %0 : i8[8,4] = [[1, 2, 3, 4], [5, 6, 7, 8], [1, 1, 1, 1], [2, 2, 2, 2], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
          host.alloc %2 : i32[4,4] = 0
          tpu.pass #0 a=%X[0:4,0:4] b=%0[0:4,0:4]  // 15 cyc
          host.acc %2[0:4,0:4] += #0 {relu}
        }
        """)
    assert comp.program["estimate"]["passes"] == 1
    assert comp.program["estimate"]["passes_skipped"] == 1
    check_against_oracle(comp)


def test_o0_keeps_every_pass():
    comp = build(ZERO_K_TILE, opt=0)
    assert comp.program["estimate"]["passes"] == 2
    assert comp.program["estimate"]["passes_skipped"] == 0
    check_against_oracle(comp)


def test_an_all_zero_constant_with_a_bias_becomes_a_standalone_epilogue():
    comp = build(
        "input X : i8[4, 4]\nconst Z : i8[4, 4] = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]\n"
        "input b : i32[4]\nY = relu(X @ Z + b)\noutput Y\n"
    )
    cmds = comp.schedule.commands
    assert not any(isinstance(c, (Pass, Acc)) for c in cmds)
    (epi,) = [c for c in cmds if isinstance(c, Epilogue)]
    assert (epi.bias, epi.relu) == ("b", True)
    check_against_oracle(comp)


def test_an_all_zero_constant_with_only_relu_needs_no_work_at_all():
    comp = build(
        "input X : i8[4, 4]\nconst Z : i8[4, 4] = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]\n"
        "Y = relu(X @ Z)\noutput Y\n"
    )
    est = estimate(comp.schedule)
    assert est["passes"] == 0 and est["host_ops"]["acc"] == 0 and est["host_ops"]["epilogue"] == 0
    check_against_oracle(comp)  # the output is all zeros, straight from the zero-initialized buffer


def test_the_zero_tile_can_be_on_the_left_operand_too():
    comp = build(
        "const Z : i8[4, 8] = [[1,2,3,4,0,0,0,0],[1,1,1,1,0,0,0,0],[2,2,2,2,0,0,0,0],[0,0,0,0,0,0,0,0]]\n"
        "input X : i8[8, 4]\nY = Z @ X\noutput Y\n"
    )
    assert comp.program["estimate"]["passes"] == 1
    check_against_oracle(comp)


def test_pass_ids_are_renumbered_contiguously_after_skipping():
    # Three K tiles, the middle one all zero.
    rows = [[1, 1, 1, 1]] * 4 + [[0, 0, 0, 0]] * 4 + [[2, 2, 2, 2]] * 4
    comp = build(f"input X : i8[4, 12]\nconst W : i8[12, 4] = {rows}\nY = X @ W\noutput Y\n")
    assert [c.id for c in comp.schedule.commands if isinstance(c, Pass)] == [0, 1]
    assert [c.pass_id for c in comp.schedule.commands if isinstance(c, Acc)] == [0, 1]
    check_against_oracle(comp)


def test_nothing_is_skipped_when_no_constant_tile_is_zero():
    comp = build("input X : i8[4, 4]\nconst W : i8[4, 4] = [[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]\n"
                 "Y = X @ W\noutput Y\n")  # fmt: skip
    assert comp.program["estimate"]["passes_skipped"] == 0  # one non-zero entry keeps the tile


@pytest.mark.parametrize("opt", [0, 1])
def test_verify_each_validates_the_tile_passes(opt):
    build(ZERO_K_TILE, opt=opt, verify_each=True)
