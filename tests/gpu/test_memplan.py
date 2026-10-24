import pytest
from hypothesis import HealthCheck, event, given, settings

from tests.gen import programs
from tests.helpers import ir
from tinyc.driver import compile_graph
from tinyc.gpu.memplan import MemPlan, Region, check_plan, plan_memory
from tinyc.ir.errors import PassError


def plan(text):
    f = ir(text)
    p = plan_memory(f)
    check_plan(f, p)
    return p


CHAIN = """
    func @main(%A: i8[2,2]) -> (i8[2,2]) {
      %0 = relu %A : i8[2,2]
      %1 = relu %0 : i8[2,2]
      %2 = relu %1 : i8[2,2]
      %3 = relu %2 : i8[2,2]
      return %3
    }
    """


def test_inputs_then_constants_are_laid_out_first_and_pinned():
    p = plan("""
        func @main(%A: i8[2,3], %b: i32[3]) -> (i32[2,3]) {
          %0 = const [[1, 2, 3], [4, 5, 6]] : i8[2,3]
          %1 = add %A, %0 : i32[2,3]
          return %1
        }
        """)
    assert (p.buffers["A"].addr, p.buffers["b"].addr, p.buffers["0"].addr) == (0, 6, 9)
    assert p.buffers["0"].kind == "const" and p.buffers["A"].kind == "input"


def test_dead_intermediates_are_reused_but_outputs_are_not():
    p = plan(CHAIN)
    # %0 and %1 cannot share (both live at op 1); %2 reuses %0's words; %3 (output) is pinned
    assert p.buffers["0"].addr == 4 and p.buffers["1"].addr == 8
    assert p.buffers["2"].addr == p.buffers["0"].addr
    assert p.mem_size == 12  # 4 input + 3 live temps... not the naive 4 + 4*4 = 20


def test_a_result_never_overlaps_what_its_own_op_reads():
    p = plan(CHAIN)
    for dst, src in (("1", "0"), ("2", "1"), ("3", "2")):
        a, b = p.buffers[dst], p.buffers[src]
        assert a.addr >= b.addr + b.size or b.addr >= a.addr + a.size


def test_an_output_that_is_an_input_or_constant_needs_no_new_memory():
    p = plan("func @main(%A: i8[3]) -> (i8[3]) {\n  return %A\n}")
    assert p.mem_size == 3


def test_check_plan_detects_overlap_of_simultaneously_live_buffers():
    f = ir(CHAIN)
    p = plan_memory(f)
    p.buffers["1"] = Region("1", p.buffers["0"].addr, 4, "i8", (2, 2), "temp")  # collide with %0
    with pytest.raises(AssertionError, match="overlap"):
        check_plan(f, p)


def test_the_memory_budget_is_enforced_by_legalization():
    from tinyc.passes import PassContext

    with pytest.raises(PassError, match="words of tiny-gpu memory"):
        compile_graph(
            "input A : i8[30, 30]\nB = relu(A)\noutput B\n", "t.tiny", target="gpu",
            ctx=PassContext(gpu_mem_words=100),
        )  # fmt: skip


def test_the_backend_checks_the_planned_size_too_when_legalization_is_disabled():
    from tinyc.driver import compile_gpu
    from tinyc.passes import PassContext

    with pytest.raises(PassError, match="words of tiny-gpu memory"):
        compile_gpu(
            "input A : i8[30, 30]\nB = relu(A)\noutput B\n", "t.tiny",
            disabled=["legalize-gpu"], ctx=PassContext(gpu_mem_words=100),
        )  # fmt: skip


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(programs())
def test_plans_for_random_programs_never_overlap_live_buffers(source):
    try:
        func, _ = compile_graph(source, "g.tiny", target="gpu")
    except PassError:
        event("rejected")
        return
    p = plan_memory(func)
    check_plan(func, p)
    assert isinstance(p, MemPlan) and p.mem_size >= sum(a.type.size for a in func.args)
