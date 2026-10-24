"""Error messages are a feature: each broken program's rendered diagnostic is a golden file."""

from pathlib import Path

import pytest

from tinyc.frontend import CompileError, compile_to_graph

GOLDEN = Path(__file__).resolve().parent.parent / "golden" / "errors"
CASES = sorted(GOLDEN.glob("*.tiny"))


def test_cases_exist():
    assert len(CASES) >= 15


@pytest.mark.parametrize("src", CASES, ids=lambda p: p.stem)
def test_diagnostic_snapshot(src, golden):
    with pytest.raises(CompileError) as exc:
        compile_to_graph(src.read_text(), src.name)
    golden(src.with_suffix(".txt"), exc.value.render() + "\n")
