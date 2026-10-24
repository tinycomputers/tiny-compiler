"""Lowering Graph IR -> Tile IR, the Tile IR verifier, and the cost model's arithmetic."""

from pathlib import Path

import pytest

from tinyc.driver import compile_graph
from tinyc.tpu import compile_tile_ir
from tinyc.tpu.cost import estimate
from tinyc.tpu.lower import lower
from tinyc.tpu.tile_ir import (
    CYCLES_PER_PASS,
    Acc,
    Buffer,
    Epilogue,
    HostAdd,
    HostRelu,
    Pass,
    Schedule,
    ScheduleError,
    TileRef,
    print_schedule,
    verify_schedule,
)

ROOT = Path(__file__).resolve().parent.parent.parent
TPU_EXAMPLES = sorted(p for p in (ROOT / "examples").glob("*.tiny") if p.stem != "mlp_gpu")


def tile(source, opt=1, **kw):
    func, _ = compile_graph(source, "t.tiny", target="tpu", opt_level=opt)
    return compile_tile_ir(func, opt, **kw)


def dense(m, k, n, body="C = A @ B", bias=False):
    extra = f"input b : i32[{n}]\n" if bias else ""
    return f"input A : i8[{m}, {k}]\ninput B : i8[{k}, {n}]\n{extra}{body}\noutput C\n"


@pytest.mark.parametrize(
    "m,k,n,passes", [(4, 4, 4, 1), (4, 8, 4, 2), (5, 6, 7, 8), (12, 12, 12, 27), (1, 1, 1, 1)]
)
def test_pass_count_is_the_product_of_tile_counts(m, k, n, passes):
    assert estimate(tile(dense(m, k, n), opt=0))["passes"] == passes


def test_tiling_order_matches_tiny_tpus_tiling_schedule():
    """Output tiles row-major, K innermost (web/src/lib/tiling.ts `tilingSchedule`)."""
    s = tile(dense(5, 6, 7), opt=0)
    got = [(p.a.r // 4, p.b.c // 4, p.a.c // 4) for p in s.commands if isinstance(p, Pass)]
    want = [(oi, oj, k) for oi in range(2) for oj in range(2) for k in range(2)]
    assert got == want
    first = next(c for c in s.commands if isinstance(c, Pass))
    assert (first.a, first.b) == (TileRef("A", 0, 0), TileRef("B", 0, 0))
    assert first.a.c == first.b.r  # A's K offset is B's row offset


def test_useful_macs_is_m_k_n_so_padding_is_the_only_waste():
    est = estimate(tile(dense(5, 6, 7), opt=0))
    assert est["useful_macs"] == 5 * 6 * 7
    assert est["capacity_macs"] == 8 * 64
    assert est["utilization"] == round(210 / 512, 4)
    assert estimate(tile(dense(12, 12, 12), opt=0))["utilization"] == 1.0


def test_epilogue_rides_on_the_last_accumulation_of_each_output_tile():
    s = tile(dense(8, 8, 4, "C = relu(A @ B + b)", bias=True), opt=1)  # -O1 fuses the epilogue
    accs = [c for c in s.commands if isinstance(c, Acc)]
    assert len(accs) == 4  # 2 output tiles x 2 K tiles
    assert [(a.bias, a.relu) for a in accs] == [(None, False), ("b", True)] * 2


def test_unfused_ops_become_whole_buffer_host_ops():
    s = tile(dense(4, 4, 4, "H = A @ B\nC = relu(H + H)"), opt=0)
    kinds = [type(c).__name__ for c in s.commands]
    assert kinds.count("HostAdd") == 1 and kinds.count("HostRelu") == 1
    assert estimate(s)["host_ops"] == {
        "alloc": 1, "acc": 1, "epilogue": 0, "host_add": 1, "host_relu": 1
    }  # fmt: skip


def test_chip_cycles_are_pass_count_times_the_measured_pass_cost():
    est = estimate(tile(dense(12, 12, 12)))
    assert est["cycles_per_pass"] == CYCLES_PER_PASS == 15
    assert est["rtl_cycles"] == 27 * 15 == 405


@pytest.mark.parametrize("path", TPU_EXAMPLES, ids=lambda p: p.stem)
def test_tile_ir_golden(path, golden):
    func, _ = compile_graph(path.read_text(), path.name, target="tpu")
    golden(
        ROOT / "tests" / "golden" / "tile" / f"{path.stem}.tile",
        print_schedule(compile_tile_ir(func)),
    )


# --- the verifier -----------------------------------------------------------------------------


def base():
    bufs = {
        "A": Buffer("A", "i8", (4, 4), "input"),
        "B": Buffer("B", "i8", (4, 4), "input"),
        "C": Buffer("C", "i32", (4, 4), "temp"),
        "v": Buffer("v", "i32", (4,), "input"),
    }
    cmds = [Pass(0, TileRef("A", 0, 0), TileRef("B", 0, 0)), Acc(0, TileRef("C", 0, 0))]
    return Schedule("main", bufs, cmds, ["A", "B", "v"], ["C"])


def test_a_lowered_schedule_verifies():
    verify_schedule(base())
    verify_schedule(lower(compile_graph(dense(5, 6, 7), "t.tiny", target="tpu")[0]))


@pytest.mark.parametrize(
    "mutate, fragment",
    [
        (
            lambda s: s.commands.__setitem__(0, Pass(0, TileRef("Z", 0, 0), TileRef("B", 0, 0))),
            "unknown buffer",
        ),
        (
            lambda s: s.commands.append(Pass(0, TileRef("A", 0, 0), TileRef("B", 0, 0))),
            "duplicate pass id",
        ),
        (lambda s: s.commands.reverse(), "not run yet"),
        (lambda s: s.commands.pop(), "never accumulated"),
        (lambda s: s.commands.append(Acc(0, TileRef("C", 0, 0))), "accumulated twice"),
        (lambda s: s.buffers["A"].__setattr__("dtype", "i32"), "must be i8"),
        (
            lambda s: s.commands.__setitem__(0, Pass(0, TileRef("A", 4, 0), TileRef("B", 0, 0))),
            "outside",
        ),
        (
            lambda s: s.commands.__setitem__(
                0, Pass(0, TileRef("A", 0, 4 - 4), TileRef("B", 4, 0))
            ),
            "outside",
        ),
        (lambda s: s.commands.append(Epilogue(TileRef("C", 0, 0), bias="A")), "rank 1"),
        (lambda s: s.commands.append(HostAdd("C", "A", "nope")), "unknown buffer"),
        (lambda s: s.commands.append(HostRelu("nope", "A")), "unknown buffer"),
        (lambda s: s.buffers["C"].__setattr__("dtype", "i8"), "must be i32"),
        (
            lambda s: s.commands.__setitem__(0, Pass(0, TileRef("A", 0, 0), TileRef("B", 1, 0))),
            "A columns start",
        ),
    ],
)
def test_verifier_rejects_malformed_schedules(mutate, fragment):
    s = base()
    mutate(s)
    with pytest.raises(ScheduleError, match=fragment):
        verify_schedule(s)
