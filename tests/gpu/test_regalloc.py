import pytest

from tests.gpu.helpers import EXAMPLES, build
from tinyc.gpu.kernel_ir import (
    Assign,
    For,
    Kernel,
    Special,
    Store,
    VInst,
    VReg,
    flatten,
)
from tinyc.gpu.regalloc import (
    RegisterPressure,
    allocatable,
    allocate,
    assert_reserved_untouched,
    live_intervals,
)

v = [VReg(i) for i in range(12)]
TID = Special("tid")


def alloc(body, regs=16):
    insts = flatten(Kernel("k", (1, 4), body, "0", "relu"))
    return insts, allocate(insts, regs, "k")


def test_the_pool_excludes_the_reserved_registers():
    assert allocatable(16) == [0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15]
    assert allocatable(8) == [0, 1, 2, 3, 4]


def test_fewer_than_8_registers_is_rejected_because_r7_holds_the_thread_id():
    with pytest.raises(ValueError, match="at least 8"):
        allocatable(7)


def test_sequential_temporaries_share_one_register():
    body = [
        Assign("set", v[0], (1,)),
        Assign("add", v[1], (v[0], 1)),  # v0 dies here, v1 may reuse its register
        Assign("add", v[2], (v[1], 1)),
        Store(TID, v[2]),
    ]
    _, a = alloc(body)
    assert a.registers_used == 1 and a.max_live == 1


def test_values_live_at_the_same_time_get_distinct_registers():
    body = [Assign("set", v[i], (i,)) for i in range(4)]
    body += [Assign("add", v[4], (v[0], v[1])), Assign("add", v[5], (v[2], v[3])),
             Assign("add", v[6], (v[4], v[5])), Store(TID, v[6])]  # fmt: skip
    _, a = alloc(body)
    assert len({a.assignment[v[i]] for i in range(4)}) == 4


def test_a_value_used_inside_a_loop_stays_live_until_the_back_edge():
    body = [
        Assign("set", v[0], (0,)),  # accumulator: defined before, read and written every iteration
        For(v[1], 0, 3, [Assign("set", v[2], (7,)), Assign("add", v[0], (v[0], v[2]))]),
        Store(TID, v[0]),
    ]
    insts = flatten(Kernel("k", (1, 4), body, "0", "relu"))
    spans = live_intervals(insts)
    back_edge = max(i for i, x in enumerate(insts) if x.op == "BRLT")
    store = max(i for i, x in enumerate(insts) if x.op == "ST")
    assert spans[v[0]][1] == store >= back_edge  # live through the loop and after it
    # the loop-local temp must not steal the accumulator's or the counter's register
    _, a = alloc(body)
    assert len({a.assignment[v[0]], a.assignment[v[1]], a.assignment[v[2]]}) == 3


def test_a_value_defined_before_a_loop_and_last_used_inside_survives_to_the_back_edge():
    body = [
        Assign("set", v[0], (5,)),  # read once per iteration, never after the loop
        Assign("set", v[3], (0,)),
        For(v[1], 0, 3, [Assign("add", v[3], (v[3], v[0]))]),
        Store(TID, v[3]),
    ]
    insts = flatten(Kernel("k", (1, 4), body, "0", "relu"))
    back_edge = max(i for i, x in enumerate(insts) if x.op == "BRLT")
    assert live_intervals(insts)[v[0]][1] == back_edge


def test_running_out_of_registers_is_an_error_that_says_how_many_were_needed():
    body = [Assign("set", v[i], (i,)) for i in range(6)]
    body += [Assign("add", v[6], (v[i], v[i + 1])) for i in range(5)]
    body += [Assign("add", v[7], (v[0], v[5])), Store(TID, v[7])]
    with pytest.raises(RegisterPressure, match=r"needs more than 5 registers.*--gpu-regs"):
        alloc(body, regs=8)
    alloc(body, regs=16)  # the same kernel fits with the default budget


def test_assertion_catches_writes_to_reserved_registers():
    insts = [VInst("SET", v[0], (1,))]
    with pytest.raises(AssertionError, match="reserved"):
        assert_reserved_untouched(insts, {v[0]: 7})
    assert_reserved_untouched(insts, {v[0]: 4})


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_real_kernels_never_use_r5_to_r7_as_destinations(path):
    import re

    for asm in build(path.read_text()).program.asm.values():
        for line in asm.splitlines():
            m = re.match(r"(SET|ADD|MUL|LD) R(\d+),", line)
            if m:
                assert int(m.group(2)) not in (5, 6, 7), line


def test_matmul_needs_only_four_registers_and_fits_the_smallest_legal_budget():
    comp = build("input A : i8[4, 8]\ninput B : i8[8, 4]\nC = relu(A @ B)\noutput C\n", regs=8)
    assert comp.program.registers == {"k0_matmul": 4}
