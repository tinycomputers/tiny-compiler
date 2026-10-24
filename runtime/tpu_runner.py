"""Run tiny-tpu passes by calling the one executor, `runtime/ttpu-cli.mjs`, through Node.

Python never re-implements the executor (PLAN §6).
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CLI = ROOT / "runtime" / "ttpu-cli.mjs"


@dataclass
class PassResult:
    c: np.ndarray  # int32 [4, 4]
    cycles: int


@dataclass
class TpuResult:
    passes: list[PassResult]
    total_cycles: int


def run_tpu_passes(passes: Sequence[tuple[np.ndarray, np.ndarray]]) -> TpuResult:
    """Run each (A, B) pair (int8, 4x4) as one pass on the real tiny-tpu RTL."""
    payload = {
        "passes": [
            {
                "a": np.asarray(a, dtype=np.int8).reshape(16).tolist(),
                "b": np.asarray(b, dtype=np.int8).reshape(16).tolist(),
            }
            for a, b in passes
        ]
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "passes.json")
        with open(path, "w") as f:
            json.dump(payload, f)
        proc = subprocess.run(
            ["node", str(CLI), path], capture_output=True, text=True, check=False, timeout=120
        )
    if proc.returncode != 0:
        raise RuntimeError(f"ttpu-cli failed ({proc.returncode}): {proc.stderr.strip()}")
    out = json.loads(proc.stdout)
    return TpuResult(
        passes=[
            PassResult(np.asarray(p["c"], dtype=np.int32).reshape(4, 4), p["cycles"])
            for p in out["passes"]
        ],
        total_cycles=out["totalCycles"],
    )
