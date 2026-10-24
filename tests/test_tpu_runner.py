"""Phase 0: drive tiny-tpu (real RTL via WASM, through Node) from pytest with no compiler yet."""

import numpy as np

from runtime.tpu_runner import run_tpu_passes

CYCLES_PER_PASS = 15  # 14 from start to done + 1 for C[3][3] to latch (PLAN §2.2)


def test_single_pass_is_bit_exact(rng):
    a = rng.integers(-128, 128, (4, 4))
    b = rng.integers(-128, 128, (4, 4))
    res = run_tpu_passes([(a, b)])
    np.testing.assert_array_equal(res.passes[0].c, a.astype(np.int64) @ b.astype(np.int64))
    assert res.passes[0].cycles == CYCLES_PER_PASS


def test_extreme_values():
    a = np.full((4, 4), -128)
    b = np.full((4, 4), -128)
    res = run_tpu_passes([(a, b)])
    assert (res.passes[0].c == 4 * 128 * 128).all()


def test_passes_are_independent_and_cost_adds_up(rng):
    # Each pass starts from psum = 0 and reloads weights, so passes never leak state.
    pairs = [(rng.integers(-128, 128, (4, 4)), rng.integers(-128, 128, (4, 4))) for _ in range(3)]
    res = run_tpu_passes(pairs)
    for (a, b), p in zip(pairs, res.passes):
        np.testing.assert_array_equal(p.c, a.astype(np.int64) @ b.astype(np.int64))
    assert res.total_cycles == 3 * CYCLES_PER_PASS


def test_host_side_k_accumulation_matches_wide_matmul(rng):
    # PLAN §2.2: accumulating over K-tiles is host work. 4x8 @ 8x4 = two passes + a host add.
    a = rng.integers(-128, 128, (4, 8))
    b = rng.integers(-128, 128, (8, 4))
    res = run_tpu_passes([(a[:, :4], b[:4, :]), (a[:, 4:], b[4:, :])])
    c = res.passes[0].c + res.passes[1].c
    np.testing.assert_array_equal(c, a.astype(np.int64) @ b.astype(np.int64))
