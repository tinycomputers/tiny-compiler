"""Pinned static cost numbers: an unexpected change in launches/instructions/loads fails CI."""

from tests.gpu.helpers import EXAMPLES, build


def test_gpu_cost_numbers_are_pinned(perf_pin):
    pinned = {}
    for opt in (0, 1):
        pinned[f"O{opt}"] = {}
        for path in EXAMPLES:
            est = build(path.read_text(), opt=opt).program.estimate
            pinned[f"O{opt}"][path.stem] = {
                k: est[k]
                for k in (
                    "launches",
                    "threads",
                    "instructions",
                    "cycles",
                    "loads",
                    "stores",
                    "registers",
                )
            }
    perf_pin("gpu", pinned)
