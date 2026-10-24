"""Phase 0: drive tiny-gpu from pytest with no compiler yet."""

from pathlib import Path

import numpy as np

from runtime.gpu_runner import run_tgpu

EXAMPLES = Path(__file__).resolve().parent.parent / "targets" / "tiny-gpu" / "examples"


def test_vector_add_matches_numpy(rng):
    n = 8
    a = rng.integers(-1000, 1000, n)
    b = rng.integers(-1000, 1000, n)
    # vector_add.tgpu reads A at 0..7, B at 8..15 and writes C at 16..23
    res = run_tgpu(
        (EXAMPLES / "vector_add.tgpu").read_text(),
        grid=(1, n),
        mem_size=64,
        memory_init={0: a, 8: b},
        num_registers=12,
    )
    assert res.read(16, n) == (a + b).tolist()


def test_counters_observe_without_changing_semantics():
    src = "SET R0, 7\nSET R1, 5\nADD R2, R0, R1\nST R7, R2\n"
    res = run_tgpu(src, grid=(1, 4), mem_size=8)
    assert res.read(0, 4) == [12] * 4
    assert dict(res.op_counts) == {"SET": 8, "ADD": 4, "ST": 4}
    assert res.instructions == 16  # thread-instructions: 4 threads x 4 instructions
    assert res.cycles == 1  # straight-line code runs in one tiny-gpu "cycle" (PLAN §2.1)


def test_instruction_table_is_restored_after_a_run():
    from tinygpu import instructions

    before = dict(instructions.INSTRUCTIONS)
    run_tgpu("SET R0, 1\n", grid=(1, 1), mem_size=4)
    assert instructions.INSTRUCTIONS == before


def test_relu_idiom_and_per_thread_branching():
    # relu via CMP/BRGT: CMP R0, 0 / BRGT skip / SET R0, 0  (tiny-gpu has no MAX)
    src = "LD R0, R7\nCMP R0, 0\nBRGT pos\nSET R0, 0\npos:\nST R7, R0\n"
    x = np.array([-5, 3, 0, 9, -1, 2], dtype=np.int64)
    res = run_tgpu(src, grid=(1, len(x)), mem_size=16, memory_init={0: x})
    assert res.read(0, len(x)) == np.maximum(x, 0).tolist()


def test_two_blocks_see_block_and_thread_ids():
    # R5 = block_id, R6 = thread_in_block; store block*10 + thread at mem[tid]
    src = "SET R0, 10\nMUL R1, R5, R0\nADD R1, R1, R6\nST R7, R1\n"
    res = run_tgpu(src, grid=(2, 3), mem_size=8)
    assert res.read(0, 6) == [0, 1, 2, 10, 11, 12]
