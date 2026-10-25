"""The browser's entry point, tested on CPython: the same code runs under Pyodide."""

import json
from pathlib import Path

import pytest

from runtime.web_api import analyze

EX = Path(__file__).resolve().parent.parent / "examples"


def run(source, opt=1, seed=42):
    return json.loads(analyze(source, opt, seed))


def test_one_call_compiles_for_both_chips_and_checks_the_gpu_against_the_oracle():
    r = run((EX / "mm_relu.tiny").read_text())
    assert r["ok"] and r["oracle"] is not None
    assert r["gpu"]["ok"] and r["gpu"]["run"]["exact"] and r["gpu"]["run"]["reservedIntact"]
    assert r["gpu"]["run"]["outputs"]["1"] == r["oracle"][0]
    assert r["tpu"]["ok"] and r["tpu"]["program"]["format"] == "ttpu"
    assert r["tpu"]["estimate"]["passes"] == 2


def test_the_gpu_estimate_bounds_contain_the_run():
    r = run((EX / "mm_bias_relu.tiny").read_text())
    est, run_ = r["gpu"]["estimate"], r["gpu"]["run"]
    assert est["instructions"]["min"] <= run_["instructions"] <= est["instructions"]["max"]
    assert est["cycles"]["min"] <= run_["cycles"] <= est["cycles"]["max"]


def test_the_chip_rejections_are_reported_per_target_not_as_a_failure():
    r = run((EX / "mlp_gpu.tiny").read_text())
    assert r["ok"] and r["gpu"]["ok"] and r["gpu"]["run"]["exact"]
    assert not r["tpu"]["ok"] and "tiny-tpu needs i8" in r["tpu"]["error"]


def test_passes_are_listed_in_order_for_the_pass_explorer():
    r = run((EX / "mm_bias_relu.tiny").read_text())
    names = [p["name"] for p in r["gpu"]["passes"]]
    assert names[0] == "input" and "fuse" in names and "range-check" in names
    assert "lower-to-kernel" in names and names[-1] == "regalloc-emit"
    fused = next(p for p in r["gpu"]["passes"] if p["name"] == "fuse")
    assert "matmul{epilogue=bias,relu}" in fused["ir"]


def test_source_errors_come_back_with_a_caret_and_no_artifacts():
    r = run("input A : i8[2, 2]\nC = A @ A +\noutput C\n")
    assert not r["ok"] and "expected an expression" in r["error"] and "^" in r["error"]


def test_an_overflow_on_the_inputs_still_compiles_and_reports_no_oracle(monkeypatch):
    # Random inputs are bounded to +-1000, so no real program overflows them; force the path.
    from tinyc.ir import ref
    from tinyc.ir.errors import RangeError

    def overflow(*_a, **_k):
        raise RangeError("%0 (matmul) does not fit in i32")

    monkeypatch.setattr(ref, "evaluate", overflow)
    r = run((EX / "mm_relu.tiny").read_text())
    assert r["ok"] and r["oracle"] is None and "overflows" in r["oracleError"]
    assert r["gpu"]["ok"] and r["gpu"]["run"] is None  # compiled, not run without an oracle
    assert r["tpu"]["ok"]


@pytest.mark.parametrize("opt", [0, 1, 2])
def test_every_level_runs_exact_on_the_gpu(opt):
    r = run((EX / "mm_relu.tiny").read_text(), opt=opt)
    assert r["gpu"]["run"]["exact"]
