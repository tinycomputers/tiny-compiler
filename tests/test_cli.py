from pathlib import Path

import pytest

from tinyc.cli import main

EX = Path(__file__).resolve().parent.parent / "examples"


def run(capsys, *args):
    code = main([*map(str, args)])
    out = capsys.readouterr()
    return code, out.out, out.err


def test_default_run_prints_optimized_ir(capsys):
    code, out, _ = run(capsys, EX / "mm_bias_relu.tiny")
    assert code == 0 and "matmul{epilogue=bias,relu}" in out


def test_o0_and_disable_keep_ops_separate(capsys):
    for flags in (["-O", "0"], ["--disable", "fuse"]):
        code, out, _ = run(capsys, EX / "mm_relu.tiny", *flags)
        assert code == 0 and "epilogue" not in out and "relu %0" in out


def test_target_tpu_rejects_chained_matmul_with_a_caret(capsys):
    code, out, err = run(capsys, EX / "mlp_gpu.tiny", "--target", "tpu")
    assert code == 1 and out == ""
    assert "mlp_gpu.tiny:7:5: error: tiny-tpu needs i8 matmul operands" in err
    assert "7 | Y = H @ W2" in err and "^" in err


def test_target_gpu_accepts_it(capsys):
    assert run(capsys, EX / "mlp_gpu.tiny", "--target", "gpu")[0] == 0


def test_print_after_and_time_passes_go_to_stderr(capsys):
    code, out, err = run(capsys, EX / "mm_relu.tiny", "--print-after", "fuse", "--time-passes")
    assert code == 0
    assert "// ---- after fuse ----" in err and "fuse " in err and "ms" in err
    assert "// ----" not in out


def test_print_after_all_lists_every_pass(capsys):
    _, _, err = run(capsys, EX / "mm_relu.tiny", "--print-after-all", "--target", "gpu")
    for name in ("input", "shape-infer", "cse", "fuse", "range-check", "legalize-gpu"):
        assert f"// ---- after {name} ----" in err


def test_verify_each_flag(capsys):
    assert run(capsys, EX / "big_mm.tiny", "--verify-each", "--target", "tpu")[0] == 0


def test_unknown_disable_is_a_usage_error(capsys):
    with pytest.raises(SystemExit) as exc:
        main([str(EX / "mm.tiny"), "--disable", "nope"])
    assert exc.value.code == 2


def test_range_failure_is_reported_at_the_source_line(tmp_path, capsys):
    src = tmp_path / "big.tiny"
    src.write_text("input a : i32[1, 4]\ninput b : i32[4, 1]\nC = a @ b\noutput C\n")
    code, _, err = run(capsys, src)
    assert code == 1 and "big.tiny:3:5: error:" in err and "does not fit in i32" in err


def test_front_end_errors_still_work(tmp_path, capsys):
    src = tmp_path / "bad.tiny"
    src.write_text("C = nope\n")
    code, _, err = run(capsys, src)
    assert code == 1 and "undefined name 'nope'" in err


# --- tiny-tpu backend ---------------------------------------------------------------------


def test_emit_tile_prints_the_schedule(capsys):
    code, out, _ = run(capsys, EX / "mm_relu.tiny", "--target", "tpu", "--emit", "tile")
    assert code == 0 and out.startswith("tpu.schedule @main") and "host.acc" in out
    assert out.count("tpu.pass") == 2


def test_emit_ttpu_prints_valid_json(capsys):
    import json

    code, out, _ = run(capsys, EX / "mm_bias_relu.tiny", "--target", "tpu", "--emit", "ttpu")
    doc = json.loads(out)
    assert code == 0 and doc["format"] == "ttpu" and doc["estimate"]["passes"] == 2


def test_stats_prints_the_cost_estimate(capsys):
    import json

    code, out, _ = run(capsys, EX / "big_mm.tiny", "--target", "tpu", "--stats")
    est = json.loads(out)
    assert code == 0 and est["passes"] == 27 and est["rtl_cycles"] == 405


def test_run_executes_on_the_real_chip_and_reports_measured_vs_estimate(capsys):
    code, out, _ = run(capsys, EX / "big_mm.tiny", "--target", "tpu", "--run")
    assert code == 0 and "bit-exact vs numpy oracle: yes" in out
    assert "measured RTL cycles: 405  (estimate 405)" in out and "not chip work" in out


def test_run_is_seeded(capsys):
    a = run(capsys, EX / "mm.tiny", "--target", "tpu", "--run", "--seed", "1")[1]
    b = run(capsys, EX / "mm.tiny", "--target", "tpu", "--run", "--seed", "1")[1]
    c = run(capsys, EX / "mm.tiny", "--target", "tpu", "--run", "--seed", "2")[1]
    assert a == b and a != c


@pytest.mark.parametrize("flags", [["--emit", "tile"], ["--emit", "ttpu"], ["--run"], ["--stats"]])
def test_backend_flags_need_a_target(flags):
    with pytest.raises(SystemExit) as exc:
        main([str(EX / "mm.tiny"), *flags])
    assert exc.value.code == 2


def test_tpu_backend_rejects_what_the_chip_cannot_run(capsys):
    code, _, err = run(capsys, EX / "mlp_gpu.tiny", "--target", "tpu", "--run")
    assert code == 1 and "tiny-tpu needs i8 matmul operands" in err


# --- tiny-gpu backend ---------------------------------------------------------------------


def test_emit_tgpu_prints_assembly_with_source_positions(capsys):
    code, out, _ = run(capsys, EX / "mm_relu.tiny", "--target", "gpu", "--emit", "tgpu")
    assert code == 0 and out.startswith("; kernel k0_matmul") and "BRLT L0" in out
    assert "; 4:5" in out


def test_emit_kernel_and_launch(capsys):
    import json

    code, out, _ = run(capsys, EX / "mlp_gpu.tiny", "--target", "gpu", "--emit", "kernel")
    assert code == 0 and out.count("kernel @") == 2 and "for v1 in 0..8" in out
    code, out, _ = run(capsys, EX / "mlp_gpu.tiny", "--target", "gpu", "--emit", "launch")
    doc = json.loads(out)
    assert code == 0 and doc["format"] == "tgpu-launch" and len(doc["kernels"]) == 2


def test_gpu_stats_prints_the_estimate_bounds(capsys):
    import json

    code, out, _ = run(capsys, EX / "mm_relu.tiny", "--target", "gpu", "--stats")
    est = json.loads(out)
    assert code == 0 and est["launches"] == 1 and est["cycles"] == {"min": 8, "max": 9}


def test_gpu_run_executes_on_the_simulator_and_reports_measured_vs_estimate(capsys):
    code, out, _ = run(capsys, EX / "mlp_gpu.tiny", "--target", "gpu", "--run")
    assert code == 0 and "bit-exact vs numpy oracle: yes" in out
    assert "R5-R7 untouched: yes" in out and "measured instructions:" in out
    assert "straight-line code in one cycle" in out


def test_the_same_program_runs_on_both_chips_from_the_same_seed(capsys):
    gpu = run(capsys, EX / "mm_relu.tiny", "--target", "gpu", "--run", "--seed", "5")[1]
    tpu = run(capsys, EX / "mm_relu.tiny", "--target", "tpu", "--run", "--seed", "5")[1]
    matrix = lambda out: out.split("bit-exact")[0]
    assert matrix(gpu) == matrix(tpu)


def test_gpu_regs_budget_and_its_error(capsys):
    # -O0 is the register-frugal code, so it fits the smallest legal register file
    base = ["--target", "gpu", "--run", "-O", "0"]
    assert run(capsys, EX / "mm_relu.tiny", *base, "--gpu-regs", "8")[0] == 0
    code, _, err = run(capsys, EX / "mm_relu.tiny", *base, "--gpu-regs", "7")
    assert code == 1 and "at least 8 registers" in err


def test_o2_tiles_into_shared_memory_and_o0_does_not(capsys):
    o2 = run(capsys, EX / "mm_relu.tiny", "--target", "gpu", "--emit", "kernel", "-O", "2")[1]
    assert "shared=8 words" in o2 and "syncb" in o2 and "shld" in o2 and "for " not in o2
    o0 = run(capsys, EX / "mm_relu.tiny", "--target", "gpu", "--emit", "kernel", "-O", "0")[1]
    assert "shared" not in o0 and "syncb" not in o0 and "for v1 in 0..8" in o0


def test_kernel_passes_can_be_disabled_by_name(capsys):
    args = [EX / "mm_relu.tiny", "--target", "gpu", "--emit", "kernel", "-O", "2"]
    no_tile = run(capsys, *args, "--disable", "shared-tile")[1]
    assert "shared" not in no_tile and "for " not in no_tile  # still unrolled
    no_unroll = run(capsys, *args, "--disable", "unroll")[1]
    assert "shared=8 words" in no_unroll and "for v1 in 0..8" in no_unroll  # still tiled


def test_o2_run_is_bit_exact_and_reports_shared_memory_use(capsys):
    code, out, _ = run(capsys, EX / "mlp_gpu.tiny", "--target", "gpu", "--run", "-O", "2")
    assert code == 0 and "bit-exact vs numpy oracle: yes" in out
    assert "shared memory: loads" in out and "barrier executions (SYNCB)" in out


def test_gpu_emit_flags_need_a_gpu_target(capsys):
    for flags in (["--emit", "tgpu"], ["--emit", "kernel", "--target", "tpu"]):
        with pytest.raises(SystemExit) as exc:
            main([str(EX / "mm.tiny"), *flags])
        assert exc.value.code == 2
