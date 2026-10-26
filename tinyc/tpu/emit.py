"""Tile IR -> `.ttpu.json` (format documented in docs/TTPU_FORMAT.md, schema in docs/ttpu.schema.json)."""

from __future__ import annotations

import json

from tinyc.tpu.cost import estimate
from tinyc.tpu.tile_ir import (
    TILE,
    Acc,
    Alloc,
    Epilogue,
    HostAdd,
    HostRelu,
    Pass,
    Schedule,
    TileRef,
)

FORMAT = "ttpu"
VERSION = 1


def _ref(r: TileRef) -> dict:
    return {"buf": r.buf, "r": r.r, "c": r.c}


def _epilogue(d: dict, bias, relu) -> dict:
    if bias is not None:
        d["bias"] = bias
    if relu:
        d["relu"] = True
    return d


def _command(c) -> dict:
    if isinstance(c, Alloc):
        return {"op": "alloc", "buf": c.buf}
    if isinstance(c, Pass):
        return {"op": "pass", "id": c.id, "a": _ref(c.a), "b": _ref(c.b)}
    if isinstance(c, Acc):
        return _epilogue({"op": "acc", "pass": c.pass_id, "dst": _ref(c.dst)}, c.bias, c.relu)
    if isinstance(c, Epilogue):
        return _epilogue({"op": "epilogue", "dst": _ref(c.dst)}, c.bias, c.relu)
    if isinstance(c, HostAdd):
        return {"op": "host_add", "dst": c.dst, "a": c.a, "b": c.b}
    assert isinstance(c, HostRelu)
    return {"op": "host_relu", "dst": c.dst, "src": c.src}


def to_json(s: Schedule) -> dict:
    buffers = {}
    for name, b in s.buffers.items():
        entry = {"dtype": b.dtype, "shape": list(b.shape), "kind": b.kind}
        if b.kind == "const":
            entry["data"] = b.data.tolist()
        buffers[name] = entry
    return {
        "format": FORMAT,
        "version": VERSION,
        "tile": TILE,
        "name": s.name,
        "buffers": buffers,
        "inputs": list(s.inputs),
        "outputs": list(s.outputs),
        "commands": [_command(c) for c in s.commands],
        "estimate": estimate(s),
    }


def dumps(program: dict) -> str:
    return json.dumps(program, indent=2) + "\n"
