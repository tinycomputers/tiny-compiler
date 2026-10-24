"""Target contract (PLAN §2): after a submodule bump these fail loudly, before anything else.

Every assertion here is a fact the compiler's design depends on.
"""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from tinygpu import instructions
from tinygpu.gpu import TinyGPU

from runtime.tpu_runner import run_tpu_passes

ROOT = Path(__file__).resolve().parent.parent


# --- tiny-gpu -----------------------------------------------------------------------------


def test_gpu_isa_is_exactly_what_the_backend_assumes():
    assert set(instructions.INSTRUCTIONS) == {
        "SET", "ADD", "MUL", "LD", "ST", "JMP", "BEQ", "BNE", "SYNC", "CSWAP",
        "CMP", "BRGT", "BRLT", "BRZ", "SHLD", "SHST", "SYNCB",
    }  # fmt: skip
    # No SUB / DIV / shifts / MAX: the backend builds relu and loops from CMP + branches.
    for missing in ("SUB", "DIV", "SHL", "SHR", "MAX"):
        assert missing not in instructions.INSTRUCTIONS


def test_gpu_reserved_registers():
    gpu = TinyGPU(num_threads=6, num_registers=16, mem_size=4)
    gpu.set_grid(2, 3)
    assert gpu.registers[:, 5].tolist() == [0, 0, 0, 1, 1, 1]  # R5 = block_id
    assert gpu.registers[:, 6].tolist() == [0, 1, 2, 0, 1, 2]  # R6 = thread_in_block
    assert gpu.registers[:, 7].tolist() == [0, 1, 2, 3, 4, 5]  # R7 = global tid


def test_gpu_int32_overflow_raises_instead_of_wrapping():
    gpu = TinyGPU(num_threads=1, num_registers=8, mem_size=4)
    with pytest.raises(OverflowError):
        gpu.registers[0, 0] = 2**31


def test_gpu_set_grid_resize_leaves_new_threads_without_tid():
    # Known tiny-gpu quirk: the runner must build the simulator at its final size.
    gpu = TinyGPU(num_threads=2, num_registers=8, mem_size=4)
    gpu.set_grid(1, 4)
    assert gpu.registers[:, 7].tolist() == [0, 1, 0, 0]


def test_gpu_global_memory_survives_across_launches():
    gpu = TinyGPU(num_threads=1, num_registers=8, mem_size=4)
    gpu.memory[2] = 99
    gpu.load_kernel([("SET", [("R", 0), 1])], grid=(1, 1))
    gpu.run_kernel(max_cycles=10)
    gpu.load_kernel([("SET", [("R", 0), 2])], grid=(1, 1))
    assert gpu.memory[2] == 99


# --- tiny-tpu -----------------------------------------------------------------------------


def test_tpu_pass_cost_and_shape():
    ident = np.eye(4, dtype=np.int64)
    res = run_tpu_passes([(ident, ident)])
    assert res.passes[0].c.shape == (4, 4)
    assert res.passes[0].cycles == 15  # 4 load + 7 stream + 3 drain + 1 final latch


def test_tpu_operands_are_truncated_to_int8(tmp_path):
    # The compiler must reject non-i8 operands itself; the hardware silently truncates.
    # Bypass the Python wrapper (which would wrap in numpy) and hand the CLI a raw 128.
    a = [0] * 16
    a[0] = 128
    b = [1 if i % 5 == 0 else 0 for i in range(16)]  # identity
    f = tmp_path / "passes.json"
    f.write_text(json.dumps({"passes": [{"a": a, "b": b}]}))
    out = subprocess.run(
        ["node", str(ROOT / "runtime" / "ttpu-cli.mjs"), str(f)],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    assert json.loads(out.stdout)["passes"][0]["c"][0] == -128


# --- pinned target commits ------------------------------------------------------------------


def test_submodules_are_pinned_and_checked_out():
    status = subprocess.run(
        ["git", "submodule", "status"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    names = {line.split()[1] for line in status}
    assert names == {"targets/tiny-gpu", "targets/tiny-tpu"}
    assert not any(line.startswith(("-", "+", "U")) for line in status), status


# --- tiny-gpu facts the GPU backend's code generation depends on (measured, PLAN §2.1) ------------


def _run(src, **kw):
    from runtime.gpu_runner import run_tgpu

    kw.setdefault("grid", (1, 3))
    kw.setdefault("mem_size", 16)
    return run_tgpu(src, **kw)


def test_gpu_a_label_starting_with_r_is_parsed_as_a_register():
    # The emitter names labels L0, L1, ...; `relu_end` would crash tiny-gpu's assembler.
    with pytest.raises(ValueError):
        _run("SET R0, 1\nBRZ relu_end\nrelu_end:\nST 0, R0\n")
    _run("SET R0, 1\nBRZ L0\nL0:\nST 0, R0\n")


def test_gpu_a_loop_of_k_iterations_costs_k_cycles():
    loop = "SET R0, 0\nL0:\nADD R0, R0, 1\nCMP R0, {k}\nBRLT L0\nST R7, R0\n"
    for k in (1, 2, 5):
        assert _run(loop.format(k=k)).cycles == k


def test_gpu_relu_idiom_costs_a_cycle_only_when_a_thread_does_not_clamp():
    relu = "LD R0, R7\nCMP R0, 0\nBRGT L0\nSET R0, 0\nL0:\nADD R2, R7, 8\nST R2, R0\n"
    clamped = _run(relu, memory_init={0: [-1, -2, -3]})
    assert (clamped.cycles, clamped.op_counts["SET"]) == (1, 3)  # every thread runs the SET
    kept = _run(relu, memory_init={0: [1, 2, 3]})
    assert (kept.cycles, kept.op_counts["SET"]) == (2, 0)  # every thread takes the skip branch
    assert _run(relu, memory_init={0: [-1, 2, -3]}).read(8, 3) == [0, 2, 0]


def test_gpu_branching_to_the_end_of_the_program_costs_a_stray_cycle():
    src = "SET R0, 1\nSET R1, 0\nCMP R0, R1\nBRGT L9\nSET R2, 5\nL9:\n"
    assert _run(src, grid=(1, 1)).cycles == 2  # why a kernel must not end with an `if`


def test_gpu_addresses_may_be_immediates_and_comments_are_allowed():
    res = _run("LD R0, 1 ; load\nADD R0, R0, 5\nST 2, R0\n", grid=(1, 1), memory_init={1: [10]})
    assert res.read(0, 4) == [0, 10, 15, 0]


def test_gpu_reserved_registers_are_reported_intact_after_a_normal_run():
    assert _run("ADD R0, R7, 1\nST R7, R0\n", grid=(2, 3)).reserved_intact
    assert not _run("SET R7, 99\n", grid=(2, 3)).reserved_intact  # the check can fail
