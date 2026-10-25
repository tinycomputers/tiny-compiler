"""Pinned static cost numbers: an unexpected change in launches/instructions/loads fails CI.

The optimization levels are pinned side by side, so a pass that stops helping (or starts hurting)
shows up as a diff here. Update deliberately with `pytest --update-golden`.
"""

from tests.gpu.helpers import EXAMPLES, build

KEYS = (
    "launches",
    "threads",
    "instructions",
    "cycles",
    "loads",
    "stores",
    "shared_loads",
    "shared_stores",
    "registers",
)


def test_gpu_cost_numbers_are_pinned(perf_pin):
    pinned = {}
    for opt in (0, 1, 2):
        pinned[f"O{opt}"] = {}
        for path in EXAMPLES:
            est = build(path.read_text(), opt=opt).program.estimate
            pinned[f"O{opt}"][path.stem] = {k: est[k] for k in KEYS}
    perf_pin("gpu", pinned)
