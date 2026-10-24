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
