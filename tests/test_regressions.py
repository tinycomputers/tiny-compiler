"""Every program in tests/regressions/ runs on every backend and level. Bugs found by the fuzzer
(or by hand) are added here as `.tiny` files, so they stay fixed."""

from pathlib import Path

import pytest

from tests.fuzz_support import check_program, pipeline_is_idempotent

CASES = sorted(Path(__file__).parent.glob("regressions/*.tiny"))


def test_the_corpus_is_not_empty():
    assert len(CASES) >= 10


@pytest.mark.parametrize("path", CASES, ids=lambda p: p.stem)
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_regression_program_agrees_with_the_oracle_everywhere(path, seed):
    ran = check_program(path.read_text(), seed)
    assert "ok" in ran.values(), f"nothing ran for {path.name}: {ran}"


@pytest.mark.parametrize("path", CASES, ids=lambda p: p.stem)
def test_regression_program_pipeline_is_idempotent(path):
    assert pipeline_is_idempotent(path.read_text())
