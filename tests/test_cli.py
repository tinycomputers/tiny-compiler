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
