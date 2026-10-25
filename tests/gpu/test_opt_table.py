"""docs/ARCHITECTURE.md quotes measured numbers; they must stay equal to what the tools measure."""

import re

from tests.gpu.helpers import ROOT
from tinyc.gpu.report import CONFIGS, PROGRAMS, opt_table_markdown

DOC = ROOT / "docs" / "ARCHITECTURE.md"


def test_the_documented_optimization_table_matches_a_fresh_measurement():
    text = DOC.read_text()
    m = re.search(r"<!-- opt-table:start -->\n(.*?)\n<!-- opt-table:end -->", text, re.DOTALL)
    assert m, "markers missing from docs/ARCHITECTURE.md"
    stale = "ARCHITECTURE.md is stale: run `python scripts/opt_table.py --write`"
    assert m.group(1).strip() == opt_table_markdown().strip(), stale


def test_every_configuration_in_the_table_is_correct_and_costs_are_ordered():
    """Measuring asserts bit-exactness; here, the claims the prose makes about the numbers."""
    from tinyc.gpu.report import measure

    src = PROGRAMS["8x8"]
    m = {label: measure(src, opt, off) for label, opt, off in CONFIGS}
    o0, licm, o1 = m["-O0 (naive)"], m["+ licm"], m["+ strength-reduce (= -O1)"]
    tile, unroll, o2 = (
        m["-O2 with shared-memory tiling only"],
        m["-O2 with unrolling + folding only"],
        m["-O2 (all passes)"],
    )
    assert o0["instructions"] > licm["instructions"] > o1["instructions"]
    assert licm["registers"] > o0["registers"] and o1["registers"] >= licm["registers"]
    assert o1["loads"] == licm["loads"] < o0["loads"] + 1  # loads are not removed by licm/sr
    assert tile["loads"] < o1["loads"] and unroll["loads"] == o1["loads"]
    assert unroll["cycles"] < o1["cycles"] and o2["cycles"] < o1["cycles"]
    assert o2["instructions"] < o1["instructions"]
    assert o2["loads"] == tile["loads"] and o2["instructions"] <= unroll["instructions"]
