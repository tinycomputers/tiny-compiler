"""Static cost for kernels, to be compared with what the simulator measures.

Each thread is evaluated *abstractly*: values that depend only on the block id, thread id and
constants are computed exactly, and anything loaded from memory is UNKNOWN. So loop trip counts,
the cooperative-copy guards of a tiled kernel and every load, store and shared access are exact.
The only uncertainty is an `if` on loaded data (relu): a thread that clamps runs one more
instruction, a thread that does not takes a skip branch, which adds a cycle. Instructions and
cycles are therefore bounds `min`/`max`; `gpu.interp` gives the exact numbers for given inputs.

A barrier is handled per block with the arrival/release model described in `gpu.interp`.
"""

from __future__ import annotations

from dataclasses import dataclass

from tinyc.gpu.kernel_ir import (
    Assign,
    Barrier,
    For,
    If,
    Kernel,
    Load,
    SharedLoad,
    SharedStore,
    Special,
    Store,
)

UNKNOWN = None  # a value that depends on loaded data


@dataclass
class _Tally:
    imin: int = 0
    imax: int = 0
    tmin: int = 0  # taken branches
    tmax: int = 0
    loads: int = 0
    stores: int = 0
    shld: int = 0
    shst: int = 0


def _value(x, env):
    if isinstance(x, Special):
        return env[x.name]
    return x if isinstance(x, int) else env.get(x)


def _dsts(body):
    for s in body:
        if isinstance(s, (Assign, Load, SharedLoad)):
            yield s.dst
        if isinstance(s, (For, If)):
            yield from _dsts(s.body)


def _eval(body: list, env: dict, t: _Tally) -> None:
    for s in body:
        if isinstance(s, Assign):
            t.imin += 1
            t.imax += 1
            a = [_value(x, env) for x in s.args]
            if any(v is UNKNOWN for v in a):
                env[s.dst] = UNKNOWN
            else:
                env[s.dst] = (
                    a[0] if s.op == "set" else a[0] + a[1] if s.op == "add" else a[0] * a[1]
                )
        elif isinstance(s, (Load, Store, SharedLoad, SharedStore)):
            t.imin += 1
            t.imax += 1
            if isinstance(s, Load):
                t.loads += 1
            elif isinstance(s, Store):
                t.stores += 1
            elif isinstance(s, SharedLoad):
                t.shld += 1
            else:
                t.shst += 1
            if isinstance(s, (Load, SharedLoad)):
                env[s.dst] = UNKNOWN
        elif isinstance(s, For):
            t.imin += 1  # SET var, start
            t.imax += 1
            env[s.var] = s.start
            trips = s.stop - s.start
            for it in range(trips):
                _eval(s.body, env, t)
                env[s.var] += 1
                t.imin += 3  # ADD, CMP, BRLT
                t.imax += 3
                if it < trips - 1:
                    t.tmin += 1
                    t.tmax += 1
        elif isinstance(s, If):
            t.imin += 2  # CMP, BRGT
            t.imax += 2
            a, b = _value(s.a, env), _value(s.b, env)
            if a is not UNKNOWN and b is not UNKNOWN:
                if a <= b:
                    _eval(s.body, env, t)
                else:
                    t.tmin += 1
                    t.tmax += 1
            else:  # decided by loaded data: bound both outcomes
                sub = _Tally()
                _eval(s.body, dict(env), sub)
                if sub.loads or sub.stores or sub.shld or sub.shst:
                    raise NotImplementedError("memory access under a data-dependent `if`")
                t.imax += sub.imax
                t.tmin += min(1, sub.tmin)
                t.tmax += max(1, sub.tmax)
                for d in _dsts(s.body):
                    env[d] = UNKNOWN
        elif isinstance(s, Barrier):
            raise TypeError("barriers are handled by kernel_cost, at the top level only")


def kernel_cost(k: Kernel, registers: int) -> dict:
    blocks, tpb = k.grid
    split = next((i for i, s in enumerate(k.body) if isinstance(s, Barrier)), None)
    before = k.body if split is None else k.body[:split]
    after = [] if split is None else k.body[split + 1 :]

    imin = imax = loads = stores = shld = shst = 0
    cyc_lo = cyc_hi = 0
    for block in range(blocks):
        p1s, p2s = [], []
        for thread in range(tpb):
            env = {"block": block, "thread": thread, "tid": block * tpb + thread}
            p1, p2 = _Tally(), _Tally()
            _eval(before, env, p1)
            _eval(after, env, p2)
            p1s.append(p1)
            p2s.append(p2)
        if split is None:
            lo = max(p.tmin for p in p1s) + 1
            hi = max(p.tmax for p in p1s) + 1
            for p in p1s:
                imin += p.imin
                imax += p.imax
        else:
            smin = [p.tmin + 1 for p in p1s]  # arrival step bounds
            smax = [p.tmax + 1 for p in p1s]
            rmin, rmax = max(smin), max(smax)  # barrier release step bounds
            lo = rmin + max(p.tmin for p in p2s) + 1
            hi = rmax + max(p.tmax for p in p2s) + 1
            for p, q, a_lo, a_hi in zip(p1s, p2s, smin, smax):
                imin += p.imin + q.imin + max(1, rmin - a_hi + 1)  # SYNCB re-executions
                imax += p.imax + q.imax + (rmax - a_lo + 1)
        cyc_lo, cyc_hi = max(cyc_lo, lo), max(cyc_hi, hi)
        for p in (*p1s, *p2s):
            loads += p.loads
            stores += p.stores
            shld += p.shld
            shst += p.shst
    return {
        "name": k.name,
        "grid": list(k.grid),
        "threads": k.threads,
        "shared_size": k.shared_size,
        "instructions": {"min": imin, "max": imax},
        "cycles": {"min": cyc_lo, "max": cyc_hi},
        "loads": loads,
        "stores": stores,
        "shared_loads": shld,
        "shared_stores": shst,
        "registers": registers,
    }


def estimate(kernels: list[Kernel], registers: dict[str, int]) -> dict:
    per = [kernel_cost(k, registers[k.name]) for k in kernels]
    return {
        "launches": len(per),
        "threads": sum(p["threads"] for p in per),
        "instructions": {
            "min": sum(p["instructions"]["min"] for p in per),
            "max": sum(p["instructions"]["max"] for p in per),
        },
        "cycles": {
            "min": sum(p["cycles"]["min"] for p in per),
            "max": sum(p["cycles"]["max"] for p in per),
        },
        "loads": sum(p["loads"] for p in per),
        "stores": sum(p["stores"] for p in per),
        "shared_loads": sum(p["shared_loads"] for p in per),
        "shared_stores": sum(p["shared_stores"] for p in per),
        "registers": max((p["registers"] for p in per), default=0),
        "per_kernel": per,
    }
