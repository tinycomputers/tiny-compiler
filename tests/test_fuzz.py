"""Differential fuzzing: random legal programs, every backend and level, real chips.

    pytest tests/test_fuzz.py                          # 100 examples, fixed seed (CI)
    FUZZ_EXAMPLES=5000 FUZZ_RANDOM=1 pytest tests/test_fuzz.py   # nightly: fresh programs

A failing program is written, shrunk by Hypothesis, to tests/regressions/_last_failure.tiny
(overwritten on each failing run, so the last one is the minimal one). Rename it and commit it as a
regression once the bug is understood.
"""

import os
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tests.fuzz_support import check_program, pipeline_is_idempotent
from tests.gen import programs, tpu_programs

EXAMPLES = int(os.environ.get("FUZZ_EXAMPLES", "100"))
RANDOM = os.environ.get("FUZZ_RANDOM") == "1"
LAST_FAILURE = Path(__file__).parent / "regressions" / "_last_failure.tiny"

FUZZ = settings(
    max_examples=EXAMPLES,
    deadline=None,
    derandomize=not RANDOM,
    database=None,
    suppress_health_check=list(HealthCheck),
)


@FUZZ
@given(st.one_of(tpu_programs(), programs()), st.integers(0, 2**32 - 1))
def test_every_backend_and_level_agrees_with_the_oracle(source, seed):
    try:
        check_program(source, seed)
        assert pipeline_is_idempotent(source)
    except BaseException:
        LAST_FAILURE.write_text(source)
        raise
