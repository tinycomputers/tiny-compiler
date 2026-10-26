"""Pinned cost numbers: an unexpected change in passes/cycles/host ops fails CI."""

from tests.tpu.helpers import TPU_EXAMPLES, build


def test_tpu_cost_numbers_are_pinned(perf_pin):
    pinned = {}
    for opt in (0, 1):
        pinned[f"O{opt}"] = {}
        for path in TPU_EXAMPLES:
            est = build(path.read_text(), opt=opt).program["estimate"]
            pinned[f"O{opt}"][path.stem] = {
                k: est[k]
                for k in ("passes", "rtl_cycles", "host_ops", "useful_macs", "utilization")
            }
    perf_pin("tpu", pinned)
