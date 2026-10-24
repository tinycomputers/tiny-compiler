"""Run tiny-tpu passes by calling the one executor, `runtime/ttpu-cli.mjs`, through Node.

Python never re-implements the executor (PLAN §6).
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
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
    out = _call(payload)
    return TpuResult(
        passes=[
            PassResult(np.asarray(p["c"], dtype=np.int32).reshape(4, 4), p["cycles"])
            for p in out["passes"]
        ],
        total_cycles=out["totalCycles"],
    )


@dataclass
class ProgramResult:
    outputs: dict[str, np.ndarray]  # int64, original shapes
    pass_cycles: list[int]  # measured per pass (RTL clock cycles)
    total_cycles: int
    host_ops: dict[str, int]  # counted by the executor, never attributed to the chip
    passes: list[dict] = field(default_factory=list)


def _call(request: dict) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "request.json")
        with open(path, "w") as f:
            json.dump(request, f)
        proc = subprocess.run(
            ["node", str(CLI), path], capture_output=True, text=True, check=False, timeout=300
        )
    if proc.returncode != 0:
        raise RuntimeError(f"ttpu-cli failed ({proc.returncode}): {proc.stderr.strip()}")
    return json.loads(proc.stdout)


def run_ttpu(program: dict, inputs: dict) -> ProgramResult:
    """Execute a `.ttpu.json` program on the real tiny-tpu RTL through the one JS executor."""
    payload = {k: np.asarray(v, dtype=np.int64).tolist() for k, v in inputs.items()}
    out = _call({"program": program, "inputs": payload})
    return ProgramResult(
        outputs={
            name: np.asarray(o["data"], dtype=np.int64).reshape(o["shape"])
            for name, o in out["outputs"].items()
        },
        pass_cycles=[p["cycles"] for p in out["passes"]],
        total_cycles=out["totalCycles"],
        host_ops=out["hostOps"],
        passes=out["passes"],
    )
