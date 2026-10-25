"""The kernel optimizations: directed tests per pass, then what they buy, measured on the simulator."""

import copy
import dataclasses
from collections import Counter

import numpy as np
import pytest

from runtime.gpu_runner import run_tgpu
from tests.gpu.helpers import EXAMPLES, ROOT, build, run_on_simulator
from tinyc.gpu import cost, interp
from tinyc.gpu.emit import emit_asm
from tinyc.gpu.kernel_ir import (
    Assign,
    Barrier,
    For,
    If,
    Kernel,
    Load,
    SharedLoad,
    SharedStore,
    Special,
    Store,
    VReg,
    flatten,
    print_kernel,
    verify_kernel,
)
from tinyc.gpu.kernel_passes import (
    KERNEL_PASSES,
    LICM,
    Fold,
    KernelDCE,
    SharedTile,
    StrengthReduce,
    Unroll,
    kernel_pass_names,
)
from tinyc.gpu.regalloc import allocate
from tinyc.ir import ref
from tinyc.ir.types import epilogue

v = [VReg(i) for i in range(16)]
BLOCK, THREAD, TID = Special("block"), Special("thread"), Special("tid")


def matmul_src(m, k, n, body="C = A @ B"):
    return f"input A : i8[{m}, {k}]\ninput B : i8[{k}, {n}]\n{body}\noutput C\n"


def lowered(source, opt=0):
    """The kernels of a program, copied so a pass can be applied by hand."""
    prog = build(source, opt=opt).program
    return prog, copy.deepcopy(prog.kernels)


def kernel(body, grid=(1, 4), shared=0):
    return Kernel("k", grid, body, "0", "matmul", None, shared)


def run_kernel(k: Kernel, memory: list[int]) -> tuple[list[int], Counter]:
    memory, ops = list(memory), Counter()
    for block in range(k.grid[0]):
        interp.run_block(k, memory, block, ops)
    return memory, ops


def outputs_of(prog, kernels, seed=1):
    inputs = ref.random_inputs(prog.func, np.random.default_rng(seed))
    return interp.run_program(dataclasses.replace(prog, kernels=kernels), inputs)["outputs"], inputs


def same_outputs(prog, kernels, seed=1):
    got, inputs = outputs_of(prog, kernels, seed)
    for name, want in zip([x.name for x in prog.func.outputs], ref.evaluate(prog.func, inputs)):
        np.testing.assert_array_equal(got[name], want)


def top_loop(k: Kernel) -> For:
    return next(s for s in k.body if isinstance(s, For))


def kinds(body) -> list[str]:
    return [s.op if isinstance(s, Assign) else type(s).__name__.lower() for s in body]


# --- levels and names -------------------------------------------------------------------------


def test_levels_select_the_documented_passes():
    assert kernel_pass_names(0) == []
    assert kernel_pass_names(1) == ["licm", "strength-reduce", "kernel-dce"]
    assert kernel_pass_names(2) == [
        "licm", "strength-reduce", "shared-tile", "unroll", "fold", "kernel-dce",
    ]  # fmt: skip
    assert set(kernel_pass_names(2)) <= set(KERNEL_PASSES)


# --- licm --------------------------------------------------------------------------------------


def test_licm_hoists_the_address_part_that_does_not_depend_on_k():
    prog, ks = lowered(matmul_src(4, 8, 4))
    (k,) = ks
    LICM().run(k)
    verify_kernel(k)
    loop = top_loop(k)
    mul_by_block = lambda s: isinstance(s, Assign) and s.op == "mul" and BLOCK in s.args
    assert not any(mul_by_block(s) for s in loop.body) and any(mul_by_block(s) for s in k.body)
    assert sum(isinstance(s, Load) for s in loop.body) == 2  # loads are never hoisted
    assert any(s.op == "mul" and loop.var in s.args for s in loop.body if isinstance(s, Assign))
    same_outputs(prog, ks)


def test_licm_leaves_multiply_assigned_values_and_loads_in_place():
    k = kernel([
        Assign("set", v[0], (0,)),
        Assign("set", v[1], (5,)),
        For(v[2], 0, 3, [
            Assign("add", v[1], (v[1], 1)),  # v1 changes every iteration: not invariant
            Assign("mul", v[3], (BLOCK, 7)),  # invariant
            Load(v[4], v[3]),
            Assign("add", v[0], (v[0], v[4])),
        ]),
        Store(TID, v[0]),
    ])  # fmt: skip
    LICM().run(k)
    loop = top_loop(k)
    assert kinds(loop.body) == ["add", "load", "add"]  # only the invariant multiply left the loop
    assert any(isinstance(s, Assign) and s.op == "mul" for s in k.body)


def test_licm_hoists_through_nested_loops():
    k = kernel([
        Assign("set", v[0], (0,)),
        For(v[1], 0, 2, [For(v[2], 0, 2, [
            Assign("mul", v[3], (BLOCK, 3)),
            Assign("add", v[0], (v[0], v[3])),
        ])]),
        Store(TID, v[0]),
    ])  # fmt: skip
    LICM().run(k)
    assert kinds(k.body)[:2] == ["set", "mul"]  # the multiply climbed out of both loops


# --- strength reduction ------------------------------------------------------------------------


def test_strength_reduce_turns_the_b_address_chain_into_a_pointer():
    prog, ks = lowered(matmul_src(4, 8, 4))
    (k,) = ks
    for p in (LICM(), StrengthReduce(), KernelDCE()):
        p.run(k)
    verify_kernel(k)
    loop = top_loop(k)
    muls = [s for s in loop.body if isinstance(s, Assign) and s.op == "mul"]
    assert len(muls) == 1  # only the product x*y is left: no `mul k, N`
    bumps = [s for s in loop.body if isinstance(s, Assign) and s.op == "add" and s.dst == s.args[0]]
    assert any(s.args[1] == 4 for s in bumps)  # the B pointer advances by N per iteration
    same_outputs(prog, ks)


def test_strength_reduce_handles_a_loop_that_does_not_start_at_zero():
    def build_kernel():
        return kernel(
            [
                Assign("set", v[0], (0,)),
                For(v[1], 2, 6, [
                    Assign("mul", v[2], (v[1], 3)),
                    Assign("add", v[3], (v[2], THREAD)),
                    Assign("add", v[4], (v[3], 100)),
                    Load(v[5], v[4]),
                    Assign("add", v[0], (v[0], v[5])),
                ]),
                Assign("add", v[6], (TID, 200)),
                Store(v[6], v[0]),
            ],
            grid=(1, 3),
        )  # fmt: skip

    memory = list(range(1000, 1300))
    before, ops_before = run_kernel(build_kernel(), memory)
    k = build_kernel()
    for p in (StrengthReduce(), KernelDCE()):
        p.run(k)
    verify_kernel(k)
    after, ops_after = run_kernel(k, memory)
    assert after == before
    assert ops_after["MUL"] < ops_before["MUL"]


def test_strength_reduce_skips_chains_that_are_read_after_the_loop_or_are_too_short():
    k = kernel([
        Assign("set", v[0], (0,)),
        For(v[1], 0, 3, [
            Assign("mul", v[2], (v[1], 3)),
            Assign("add", v[3], (v[2], THREAD)),
            Assign("add", v[0], (v[0], v[3])),
        ]),
        Store(TID, v[3]),  # v3 is read after the loop: it must keep its per-iteration meaning
    ])  # fmt: skip
    before = print_kernel(k)
    StrengthReduce().run(k)
    assert print_kernel(k) == before
    short = kernel([
        Assign("set", v[0], (0,)),
        For(v[1], 0, 3, [Assign("add", v[2], (v[1], 9)), Assign("add", v[0], (v[0], v[2]))]),
        Store(TID, v[0]),
    ])  # fmt: skip
    before = print_kernel(short)
    StrengthReduce().run(short)
    assert print_kernel(short) == before  # one add costs the same as the increment: no gain


# --- dead code elimination ---------------------------------------------------------------------


def test_kernel_dce_removes_dead_chains_and_keeps_what_reaches_a_store():
    k = kernel([
        Assign("set", v[0], (1,)),
        Assign("add", v[1], (v[0], 2)),
        Assign("add", v[2], (v[1], 3)),  # dead chain
        Assign("set", v[3], (9,)),
        Load(v[4], TID),  # dead load
        Store(TID, v[3]),
    ])  # fmt: skip
    KernelDCE().run(k)
    assert kinds(k.body) == ["set", "store"]


def test_kernel_dce_removes_a_dead_value_that_only_updates_itself():
    k = kernel([
        Assign("set", v[0], (0,)),
        For(v[1], 0, 3, [Assign("add", v[0], (v[0], 1))]),  # v0 is never read by anything live
        Store(TID, 5),
    ])  # fmt: skip
    KernelDCE().run(k)
    assert isinstance(k.body[0], For) and k.body[0].body == []


# --- shared-memory tiling ----------------------------------------------------------------------


def tiled(m, k, n):
    prog, ks = lowered(matmul_src(m, k, n), opt=1)
    (kern,) = ks
    SharedTile().run(kern)
    verify_kernel(kern)
    return prog, ks, kern


def test_shared_tile_copies_the_a_row_once_per_block_and_reads_it_with_shld():
    prog, ks, k = tiled(4, 8, 4)
    assert k.shared_size == 8
    assert isinstance(
        k.body[[type(s) for s in k.body].index(Barrier) + 1], For
    )  # barrier, then loop
    loop = top_loop(k)
    assert sum(isinstance(s, SharedLoad) for s in loop.body) == 1
    assert sum(isinstance(s, Load) for s in loop.body) == 1  # only B is still loaded from global
    same_outputs(prog, ks)


@pytest.mark.parametrize(
    "m,k,n,guards", [(4, 8, 4, 0), (4, 6, 4, 1), (3, 3, 4, 1), (2, 5, 2, 1), (2, 8, 2, 0)]
)
def test_the_cooperative_copy_guards_only_the_rounds_that_can_run_past_the_row(m, k, n, guards):
    prog, ks, kern = tiled(m, k, n)
    assert sum(isinstance(s, If) for s in kern.body) == guards
    stores = [s for s in kern.body if isinstance(s, SharedStore)]
    stores += [
        t for s in kern.body if isinstance(s, If) for t in s.body if isinstance(t, SharedStore)
    ]
    assert len(stores) == -(-k // n)  # ceil(K / N) rounds of one shared store each
    same_outputs(prog, ks)


def test_shared_tile_skips_one_thread_blocks_and_kernels_without_a_uniform_load():
    _, ks = lowered(matmul_src(4, 8, 1), opt=1)  # N = 1: a block is a single thread
    SharedTile().run(ks[0])
    assert ks[0].shared_size == 0 and not any(isinstance(s, Barrier) for s in ks[0].body)
    _prog, ks = lowered("input a : i8[8]\ninput b : i8[8]\nC = a + b\noutput C\n", opt=1)
    SharedTile().run(ks[0])
    assert ks[0].shared_size == 0


def test_a_tiled_kernel_does_k_global_loads_per_block_instead_of_n_times_k():
    _prog, _ks, k = tiled(4, 8, 4)
    _, ops = run_kernel(k, [1] * 400)  # data does not matter for counts here
    blocks, n, kk = 4, 4, 8
    assert ops["LD"] == blocks * kk + blocks * n * kk  # A row once per block + B per thread
    assert ops["SHST"] == blocks * kk and ops["SHLD"] == blocks * n * kk


# --- unrolling ---------------------------------------------------------------------------------


def test_unroll_replaces_a_short_loop_by_straight_line_copies_with_constant_counters():
    prog, ks = lowered(matmul_src(2, 4, 2), opt=1)
    (k,) = ks
    Unroll().run(k)
    verify_kernel(k)
    assert not any(isinstance(s, For) for s in k.body)
    assert sum(isinstance(s, Load) for s in k.body) == 2 * 4  # two loads per unrolled iteration
    same_outputs(prog, ks)


def test_unroll_leaves_long_loops_and_loops_whose_counter_is_read_later():
    _prog, ks = lowered(matmul_src(2, 17, 2), opt=1)  # 17 iterations > 16
    Unroll().run(ks[0])
    assert any(isinstance(s, For) for s in ks[0].body)
    after_use = kernel([
        Assign("set", v[0], (0,)),
        For(v[1], 0, 3, [Assign("add", v[0], (v[0], 1))]),
        Store(TID, v[1]),  # the counter's final value is read
    ])  # fmt: skip
    Unroll().run(after_use)
    assert isinstance(after_use.body[1], For)


# --- constant folding --------------------------------------------------------------------------


def test_fold_propagates_constants_and_reassociates_add_chains():
    k = kernel([
        Assign("set", v[0], (3,)),
        Assign("add", v[1], (v[0], 4)),
        Assign("add", v[2], (v[1], THREAD)),
        Assign("add", v[3], (v[2], 10)),
        Store(v[3], 1),
    ])  # fmt: skip
    Fold().run(k)
    KernelDCE().run(k)
    verify_kernel(k)
    assert [type(s).__name__ for s in k.body] == ["Assign", "Store"]
    assert k.body[0].args == (THREAD, 17)  # (3 + 4 + thread) + 10 = thread + 17


def test_fold_copy_propagates_adds_of_zero():
    k = kernel([
        Assign("add", v[0], (THREAD, 0)),  # a copy of the thread id
        Assign("add", v[1], (v[0], 0)),  # a copy of the copy
        Assign("add", v[2], (v[1], 5)),
        Store(v[2], 1),
    ])  # fmt: skip
    Fold().run(k)
    KernelDCE().run(k)
    assert [type(s).__name__ for s in k.body] == ["Assign", "Store"]
    assert k.body[0].args == (THREAD, 5)


def test_fold_does_not_copy_propagate_a_value_that_changes():
    k = kernel([
        Assign("set", v[0], (1,)),
        Assign("add", v[1], (v[0], 0)),  # copy of v0 as it is *now*
        Assign("add", v[0], (v[0], 10)),  # v0 changes afterwards
        Assign("add", v[2], (v[1], TID)),
        Store(v[2], v[1]),
    ], grid=(1, 4))  # fmt: skip
    memory = [0] * 32
    before, _ = run_kernel(k, memory)
    Fold().run(k)
    after, _ = run_kernel(k, memory)
    assert after == before


def test_fold_does_not_reassociate_through_a_value_that_changes():
    k = kernel([
        Assign("set", v[0], (0,)),
        Assign("add", v[1], (v[0], 1)),
        Assign("add", v[0], (v[0], 5)),  # v0 changes between v1's definition and v2's
        Assign("add", v[2], (v[1], 2)),
        Store(TID, v[2]),
    ])  # fmt: skip
    memory = [0] * 8
    before, _ = run_kernel(k, memory)
    Fold().run(k)
    after, _ = run_kernel(k, memory)
    assert after == before and k.body[3].args == (v[1], 2)


# --- the barrier model, against the real simulator ----------------------------------------------


def barrier_kernel(divergent_after: bool) -> Kernel:
    body = [
        If(THREAD, 1, [Load(v[0], TID), SharedStore(THREAD, v[0])]),  # threads 0,1 copy; 2,3 skip
        Barrier(),
        SharedLoad(v[1], 0),
        SharedLoad(v[2], 1),
        Assign("add", v[3], (v[1], v[2])),
    ]
    if divergent_after:
        body.append(If(v[3], 100, [Assign("add", v[3], (v[3], 1000))]))  # sum <= 100: some threads
    body += [Assign("add", v[4], (TID, 8)), Store(v[4], v[3])]
    return Kernel("k", (2, 4), body, "0", "matmul", None, 4)


@pytest.mark.parametrize("divergent_after", [False, True])
def test_the_barrier_model_matches_the_simulator_exactly_when_arrival_times_differ(divergent_after):
    k = barrier_kernel(divergent_after)
    verify_kernel(k)
    memory = [10, 20, 0, 0, 30, 130, 0, 0] + [0] * 8
    expected_mem, ops = run_kernel(k, memory)

    insts = flatten(k)
    asm = emit_asm(k, insts, allocate(insts, 16, "k"))
    sim = run_tgpu(
        asm, grid=k.grid, mem_size=16, memory_init={0: memory}, shared_size=k.shared_size
    )
    assert sim.read(0, 16) == expected_mem
    assert dict(sim.op_counts) == dict(ops)  # includes the SYNCB re-executions of waiting threads
    assert sim.reserved_intact

    # threads 0,1 arrive at step 1, threads 2,3 (taken skip branch) at step 2; release at step 2
    assert ops["SYNCB"] == 2 * (2 + 2 + 1 + 1)
    interp_cycles = max(interp.run_block(k, list(memory), b, Counter()) for b in range(2))
    assert sim.cycles == interp_cycles

    est = cost.kernel_cost(k, 0)  # the abstract model must agree too (nothing is data dependent
    if not divergent_after:  # except the second `if`, which depends on loaded data)
        assert est["instructions"]["min"] == est["instructions"]["max"] == sim.instructions
        assert est["cycles"]["min"] == est["cycles"]["max"] == sim.cycles
    else:
        assert est["instructions"]["min"] <= sim.instructions <= est["instructions"]["max"]
        assert est["cycles"]["min"] <= sim.cycles <= est["cycles"]["max"]


def test_a_barrier_inside_a_loop_or_if_is_rejected():
    from tinyc.gpu.kernel_ir import KernelError

    with pytest.raises(KernelError, match="barrier inside"):
        verify_kernel(kernel([For(v[0], 0, 2, [Barrier()]), Store(TID, 1)], shared=1))
    with pytest.raises(KernelError, match="shared memory used but shared_size is 0"):
        verify_kernel(kernel([SharedStore(0, 1), Store(TID, 1)]))
    with pytest.raises(KernelError, match="shared address 5 outside"):
        verify_kernel(kernel([SharedStore(5, 1), Store(TID, 1)], shared=2))


# --- what the optimizations buy, measured on the simulator ---------------------------------------


def matmul_loads(comp):
    """Expected global loads per matmul kernel at -O0/-O1, and with shared tiling at -O2."""
    out = {}
    for op in comp.func.ops:
        if op.kind != "matmul":
            continue
        (m, k), n = op.operands[0].type.shape, op.operands[1].type.shape[1]
        bias = m * n if "bias" in epilogue(op.attrs) else 0
        out[op.result.name] = (2 * m * n * k + bias, m * k + m * n * k + bias, m, n, k)
    return out


def non_matmul_loads(comp):
    """Loads of add/relu kernels, which no level changes: 2 per add thread, 1 per relu thread."""
    per_kernel = comp.program.estimate["per_kernel"]
    return sum(p["loads"] for p, k in zip(per_kernel, comp.program.kernels) if k.kind != "matmul")


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_o2_has_strictly_fewer_global_loads_than_o0_on_every_matmul(path):
    """The Phase 5 exit criterion, with the exact formula: 2*M*N*K -> M*K + M*N*K per matmul."""
    comps = {o: build(path.read_text(), opt=o) for o in (0, 1, 2)}
    measured = {o: run_on_simulator(c, 4)[0].op_counts["LD"] for o, c in comps.items()}
    for o, comp in comps.items():
        assert measured[o] == comp.program.estimate["loads"]  # the estimate is exact for loads
    for level, column in ((0, 0), (1, 0), (2, 1)):  # naive formula at -O0/-O1, tiled at -O2
        comp = comps[level]
        expected = non_matmul_loads(comp) + sum(e[column] for e in matmul_loads(comp).values())
        assert measured[level] == expected
    # hoisting and strength reduction never remove or add a load (same graph, passes switched off)
    plain = build(path.read_text(), opt=1, disabled=["licm", "strength-reduce"])
    assert run_on_simulator(plain, 4)[0].op_counts["LD"] == measured[1]
    assert measured[2] < measured[0]


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_each_level_does_less_work_than_the_one_before(path):
    r = {o: run_on_simulator(build(path.read_text(), opt=o), 6)[0] for o in (0, 1, 2)}
    assert r[1].instructions < r[0].instructions  # hoisting and pointer bumping remove work
    assert r[1].cycles <= r[0].cycles  # same trip count, plus fusion's saved launches
    assert r[2].cycles <= r[1].cycles  # no taken back edges at all


@pytest.mark.parametrize("opt", [1, 2])
def test_optimized_and_naive_kernels_compute_the_same_thing(opt):
    for path in EXAMPLES:
        naive, fast = build(path.read_text(), opt=0), build(path.read_text(), opt=opt)
        inputs = ref.random_inputs(naive.func, np.random.default_rng(13))
        a = interp.run_program(naive.program, inputs)["outputs"]
        b = interp.run_program(fast.program, inputs)["outputs"]
        for n0, n1 in zip(naive.launch["outputs"], fast.launch["outputs"]):
            np.testing.assert_array_equal(a[n0], b[n1])


def test_the_big_matmul_is_unrolled_only_when_the_loop_is_short_enough():
    big = build((ROOT / "examples" / "big_mm.tiny").read_text(), opt=2)  # K = 12 <= 16
    assert "BRLT" not in big.program.asm_text()
    long_k = build(matmul_src(2, 20, 3), opt=2)  # K = 20 > 16: loop stays, A is still tiled
    assert "BRLT" in long_k.program.asm_text() and "SYNCB" in long_k.program.asm_text()
    res, _, want = run_on_simulator(long_k, 2)
    np.testing.assert_array_equal(res.outputs["0"], want[0])
