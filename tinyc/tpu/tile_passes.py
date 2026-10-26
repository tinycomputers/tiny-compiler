"""Optimizations on the Tile IR."""

from __future__ import annotations

import numpy as np

from tinyc.tpu.tile_ir import TILE, Acc, Epilogue, Pass, Schedule, TileRef


def const_tile(s: Schedule, ref: TileRef) -> np.ndarray | None:
    """The zero-padded 4x4 window of a constant buffer, or None if the buffer is not constant."""
    b = s.buffers[ref.buf]
    if b.kind != "const":
        return None
    out = np.zeros((TILE, TILE), dtype=np.int64)
    sub = b.data[ref.r : ref.r + TILE, ref.c : ref.c + TILE]
    out[: sub.shape[0], : sub.shape[1]] = sub
    return out


class SkipZeroPasses:
    """Drop passes whose A or B tile is a constant all-zero tile: their product is zero.

    The matching accumulation goes too. If it carried the output tile's epilogue, the epilogue
    moves to the previous kept accumulation of that tile; if there is none it becomes a
    standalone epilogue (or disappears when it is relu only, since relu of the zero-initialized
    tile is a no-op). Pass ids are renumbered afterwards.
    """

    name = "skip-zero-passes"

    def run(self, s: Schedule) -> None:
        zero = set()
        for cmd in s.commands:
            if isinstance(cmd, Pass):
                for ref in (cmd.a, cmd.b):
                    t = const_tile(s, ref)
                    if t is not None and not t.any():
                        zero.add(cmd.id)
        if not zero:
            return

        out: list = []
        for cmd in s.commands:
            if isinstance(cmd, Pass) and cmd.id in zero:
                continue
            if isinstance(cmd, Acc) and cmd.pass_id in zero:
                if cmd.bias is not None or cmd.relu:
                    earlier = next(
                        (c for c in reversed(out) if isinstance(c, Acc) and c.dst == cmd.dst), None
                    )
                    if earlier is not None:
                        earlier.bias, earlier.relu = cmd.bias, cmd.relu
                    elif cmd.bias is not None:
                        out.append(Epilogue(cmd.dst, cmd.bias, cmd.relu))
                continue
            out.append(cmd)

        renumber = {c.id: i for i, c in enumerate(c for c in out if isinstance(c, Pass))}
        for c in out:
            if isinstance(c, Pass):
                c.id = renumber[c.id]
            elif isinstance(c, Acc):
                c.pass_id = renumber[c.pass_id]
        s.commands = out
        s.meta["passes_skipped"] = s.meta.get("passes_skipped", 0) + len(zero)


TILE_PASSES = {SkipZeroPasses.name: SkipZeroPasses}
