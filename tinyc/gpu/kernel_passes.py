"""Optimizations on the Kernel IR (the straightforward -O0 code is the baseline they improve).

    -O0  nothing
    -O1  licm, strength-reduce, kernel-dce
    -O2  licm, strength-reduce, shared-tile, unroll, fold, kernel-dce

Soundness rests on one convention: the kernels `lower.py` builds assign every temporary exactly
once. Only accumulators, induction variables and loop counters are assigned repeatedly, so a pass
may move or rewrite a single-assignment value freely, and must treat a multi-assignment one as
changing over time. Every pass keeps `verify_kernel` happy, and `--verify-each` re-runs the
kernels against the numpy oracle after each one.
"""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Callable

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
    VReg,
)

I32_LO, I32_HI = -(2**31), 2**31 - 1
MAX_UNROLL = 16  # full unroll when the trip count is at most this (plan §4.3)


# --- helpers ----------------------------------------------------------------------------------


def _walk(body: list):
    """Every statement, depth first."""
    for s in body:
        yield s
        if isinstance(s, (For, If)):
            yield from _walk(s.body)


def _dst(s):
    return s.dst if isinstance(s, (Assign, Load, SharedLoad)) else None


def _reads(s) -> list:
    """Operands a statement reads (not its destination)."""
    if isinstance(s, Assign):
        return list(s.args)
    if isinstance(s, Load):
        return [s.addr]
    if isinstance(s, Store):
        return [s.addr, s.src]
    if isinstance(s, SharedLoad):
        return [s.saddr]
    if isinstance(s, SharedStore):
        return [s.saddr, s.src]
    if isinstance(s, If):
        return [s.a, s.b]
    return []


def _def_counts(kernel: Kernel) -> Counter:
    """How many statements assign each virtual register. A For counts twice (var is updated)."""
    counts: Counter = Counter()
    for s in _walk(kernel.body):
        if _dst(s) is not None:
            counts[_dst(s)] += 1
        elif isinstance(s, For):
            counts[s.var] += 2
    return counts


def _max_vreg(kernel: Kernel) -> int:
    top = -1
    for s in _walk(kernel.body):
        for x in [*_reads(s), _dst(s), getattr(s, "var", None)]:
            if isinstance(x, VReg):
                top = max(top, x.id)
    return top


class _Fresh:
    def __init__(self, kernel: Kernel):
        self.next = _max_vreg(kernel) + 1

    def __call__(self) -> VReg:
        v = VReg(self.next)
        self.next += 1
        return v


def _inside_defs(loop: For) -> set[VReg]:
    out = {loop.var}
    for s in _walk(loop.body):
        if _dst(s) is not None:
            out.add(_dst(s))
        elif isinstance(s, For):
            out.add(s.var)
    return out


def _invariant(x, inside: set[VReg]) -> bool:
    return not isinstance(x, VReg) or x not in inside


def _uses_outside(kernel: Kernel, loop: For, v: VReg) -> int:
    """Reads of `v` in statements that are not part of `loop`."""

    def count(body) -> int:
        n = 0
        for s in body:
            if s is loop:
                continue
            n += sum(1 for x in _reads(s) if x == v)
            if isinstance(s, (For, If)):
                n += count(s.body)
        return n

    return count(kernel.body)


def _rewrite_reads(s, fn: Callable) -> None:
    """Replace every operand a statement reads with fn(operand), in place (recursing)."""
    if isinstance(s, Assign):
        s.args = tuple(fn(a) for a in s.args)
    elif isinstance(s, Load):
        s.addr = fn(s.addr)
    elif isinstance(s, Store):
        s.addr, s.src = fn(s.addr), fn(s.src)
    elif isinstance(s, SharedLoad):
        s.saddr = fn(s.saddr)
    elif isinstance(s, SharedStore):
        s.saddr, s.src = fn(s.saddr), fn(s.src)
    elif isinstance(s, If):
        s.a, s.b = fn(s.a), fn(s.b)
    if isinstance(s, (For, If)):
        for c in s.body:
            _rewrite_reads(c, fn)


def _apply(op: str, a: int, b: int | None = None) -> int:
    return a if op == "set" else a + b if op == "add" else a * b


# --- loop-invariant code motion ---------------------------------------------------------------


class LICM:
    """Hoist single-assignment arithmetic whose operands do not change inside a loop.

    Loads are never hoisted (no alias analysis). Loops run at least once, so hoisting an
    assignment that would have run anyway never adds an overflow or a fault.
    """

    name = "licm"

    def run(self, kernel: Kernel) -> None:
        defc = _def_counts(kernel)

        def go(body: list) -> list:
            out: list = []
            for s in body:
                if isinstance(s, For):
                    s.body = go(s.body)  # inner loops first: their hoists can climb further
                    inside = _inside_defs(s)
                    hoisted, kept = [], []
                    for t in s.body:
                        if (
                            isinstance(t, Assign)
                            and defc[t.dst] == 1
                            and all(_invariant(a, inside) for a in t.args)
                        ):
                            hoisted.append(t)
                            inside.discard(t.dst)
                        else:
                            kept.append(t)
                    s.body = kept
                    out.extend(hoisted)
                elif isinstance(s, If):
                    s.body = go(s.body)
                out.append(s)
            return out

        kernel.body = go(kernel.body)


# --- strength reduction ------------------------------------------------------------------------


def _fold_add(a, b, pre: list, fresh: _Fresh, loc=None):
    if isinstance(a, int) and isinstance(b, int):
        return a + b
    if isinstance(a, int) and a == 0:
        return b
    if isinstance(b, int) and b == 0:
        return a
    v = fresh()
    pre.append(Assign("add", v, (a, b), loc))
    return v


def _fold_mul(a, c: int, pre: list, fresh: _Fresh, loc=None):
    if isinstance(a, int):
        return a * c
    if c == 0:
        return 0
    if c == 1:
        return a
    v = fresh()
    pre.append(Assign("mul", v, (a, c), loc))
    return v


class StrengthReduce:
    """Replace a multiply/add chain on a loop counter by one pointer bumped each iteration.

        t4 = mul k, N ; t5 = add t4, thread ; t6 = add t5, base ; x = ld [t6]
    becomes, with `t6 = add thread, base` computed once before the loop,
        x = ld [t6] ; ... ; t6 = add t6, N

    The chain's value is linear in k, so it can be advanced by its step instead of recomputed.
    Only chains of two or more statements are reduced (a single add costs the same as the
    increment that would replace it), only the last value of a chain is turned into a pointer,
    and only when it is not read outside the loop. The dead rest of the chain is removed by
    `kernel-dce`.
    """

    name = "strength-reduce"

    def run(self, kernel: Kernel) -> None:
        defc = _def_counts(kernel)
        fresh = _Fresh(kernel)

        def go(body: list) -> list:
            out: list = []
            for s in body:
                if isinstance(s, For):
                    s.body = go(s.body)
                    out.extend(self._reduce(kernel, s, defc, fresh))
                elif isinstance(s, If):
                    s.body = go(s.body)
                out.append(s)
            return out

        kernel.body = go(kernel.body)

    def _reduce(self, kernel: Kernel, loop: For, defc: Counter, fresh: _Fresh) -> list:
        inside = _inside_defs(loop)
        # vreg -> (step per iteration, parent IV, defining statement, the other operand)
        iv: dict[VReg, tuple] = {loop.var: (1, None, None, None)}
        for s in loop.body:
            if not isinstance(s, Assign) or defc[s.dst] != 1 or s.op == "set":
                continue
            a, b = s.args
            for x, y in ((a, b), (b, a)):
                if not (isinstance(x, VReg) and x in iv):
                    continue
                if s.op == "mul" and isinstance(y, int):
                    iv[s.dst] = (iv[x][0] * y, x, s, y)
                    break
                if s.op == "add" and _invariant(y, inside):
                    iv[s.dst] = (iv[x][0], x, s, y)
                    break

        def chain_len(v: VReg) -> int:
            return 0 if v == loop.var else 1 + chain_len(iv[v][1])

        parents = {p for (_, p, _, _) in iv.values() if p is not None}
        pre: list = []
        for d in [v for v in iv if v != loop.var]:
            if chain_len(d) < 2 or d in parents or _uses_outside(kernel, loop, d):
                continue
            if not any(d in _reads(t) for t in _walk(loop.body)):
                continue
            step = iv[d][0]

            def init(v: VReg):
                if v == loop.var:
                    return loop.start
                _, parent, stmt, other = iv[v]
                p = init(parent)
                if stmt.op == "mul":
                    return _fold_mul(p, other, pre, fresh, stmt.loc)
                return _fold_add(p, other, pre, fresh, stmt.loc)

            before = len(pre)
            value = init(d)
            if len(pre) > before and pre[-1].dst == value:
                pre[-1].dst = d  # the last computation of the init chain defines the pointer
            elif isinstance(value, int):
                pre.append(Assign("set", d, (value,), iv[d][2].loc))
            else:
                pre.append(Assign("add", d, (value, 0), iv[d][2].loc))
            loop.body = [t for t in loop.body if t is not iv[d][2]]
            loop.body.append(Assign("add", d, (d, step), iv[d][2].loc))
        return pre


# --- dead code elimination ---------------------------------------------------------------------


class KernelDCE:
    """Remove assignments and loads whose value never reaches a store, `if` or loop control."""

    name = "kernel-dce"

    def run(self, kernel: Kernel) -> None:
        live: set[VReg] = set()
        for s in _walk(kernel.body):
            if isinstance(s, (Store, SharedStore, If)):
                live.update(x for x in _reads(s) if isinstance(x, VReg))
            elif isinstance(s, For):
                live.add(s.var)
        changed = True
        while changed:
            changed = False
            for s in _walk(kernel.body):
                d = _dst(s)
                if d is not None and d in live:
                    for x in _reads(s):
                        if isinstance(x, VReg) and x not in live:
                            live.add(x)
                            changed = True

        def prune(body: list) -> list:
            out = []
            for s in body:
                if _dst(s) is not None and _dst(s) not in live:
                    continue
                if isinstance(s, (For, If)):
                    s.body = prune(s.body)
                out.append(s)
            return out

        kernel.body = prune(kernel.body)


# --- shared-memory tiling ----------------------------------------------------------------------


def _thread_deps(kernel: Kernel) -> dict[VReg, set[str]]:
    """What each vreg may depend on besides `block` and constants: 'thread' or loaded 'mem'."""
    deps: dict[VReg, set[str]] = {}
    changed = True
    while changed:
        changed = False
        for s in _walk(kernel.body):
            d = _dst(s)
            if d is None:
                continue
            new: set[str] = set()
            if isinstance(s, (Load, SharedLoad)):
                new.add("mem")
            for x in _reads(s):
                if isinstance(x, Special) and x.name in ("thread", "tid"):
                    new.add("thread")
                elif isinstance(x, VReg):
                    new |= deps.get(x, set())
            if not new <= deps.get(d, set()):
                deps.setdefault(d, set()).update(new)
                changed = True
    return deps


class SharedTile:
    """Load block-uniform data once per block into shared memory instead of once per thread.

    In a loop `for k in 0..K`, a load whose address is `u + k` with `u` the same for every
    thread of a block (it depends only on the block id and constants) reads the same K words in
    every thread. The block's threads copy those words cooperatively (thread j copies words
    j, j+N, ...), meet at a barrier, and the loop then reads `shld [k]`. Global loads drop from
    N*K to K per block. Skipped for one-thread blocks, where there is nothing to share.
    """

    name = "shared-tile"

    def run(self, kernel: Kernel) -> None:
        n = kernel.grid[1]
        if n < 2 or kernel.shared_size:
            return
        defc = _def_counts(kernel)
        deps = _thread_deps(kernel)
        fresh = _Fresh(kernel)
        uses = Counter()
        for s in _walk(kernel.body):
            uses.update(x for x in _reads(s) if isinstance(x, VReg))

        for pos, loop in enumerate(kernel.body):
            if not (isinstance(loop, For) and loop.start == 0):
                continue
            inside = _inside_defs(loop)
            addr_def = {
                t.dst: t
                for t in loop.body
                if isinstance(t, Assign) and t.op == "add" and defc[t.dst] == 1
            }
            tiles = []  # (load, address statement, u)
            for t in loop.body:
                if not (isinstance(t, Load) and isinstance(t.addr, VReg)):
                    continue
                a = addr_def.get(t.addr)
                if a is None or uses[t.addr] != 1:
                    continue
                for x, u in ((a.args[0], a.args[1]), (a.args[1], a.args[0])):
                    if (
                        x == loop.var
                        and isinstance(u, VReg)
                        and u not in inside
                        and defc[u] == 1
                        and not deps.get(u)
                    ):
                        tiles.append((t, a, u))
                        break
            if not tiles:
                continue

            k_words, pre, offset = loop.stop, [], 0
            for load, addr_stmt, u in tiles:
                for r in range((k_words + n - 1) // n):
                    idx = Special("thread") if r == 0 else fresh()
                    if r:
                        pre.append(Assign("add", idx, (Special("thread"), r * n), load.loc))
                    addr, tmp, sidx = fresh(), fresh(), idx
                    body = [Assign("add", addr, (u, idx), load.loc), Load(tmp, addr, load.loc)]
                    if offset:
                        sidx = fresh()
                        body.append(Assign("add", sidx, (idx, offset), load.loc))
                    body.append(SharedStore(sidx, tmp, load.loc))
                    if (r + 1) * n <= k_words:  # every thread's index is in range: no guard
                        pre.extend(body)
                    else:
                        pre.append(If(idx, k_words - 1, body, load.loc))
                # in the loop: the copied word for iteration k is at shared[offset + k]
                replacement = [SharedLoad(load.dst, loop.var, load.loc)]
                if offset:
                    s_k = fresh()
                    replacement = [
                        Assign("add", s_k, (loop.var, offset), load.loc),
                        SharedLoad(load.dst, s_k, load.loc),
                    ]
                new_body: list = []
                for t in loop.body:  # identity, not equality: equal-looking statements may differ
                    if t is load:
                        new_body.extend(replacement)
                    elif t is not addr_stmt:
                        new_body.append(t)
                loop.body = new_body
                offset += k_words
            pre.append(Barrier(loop.loc))
            kernel.shared_size = offset
            kernel.body[pos:pos] = pre
            return  # one loop per kernel is enough for matmul


# --- full unrolling ----------------------------------------------------------------------------


class Unroll:
    """Fully unroll innermost loops with at most MAX_UNROLL iterations.

    The counter becomes a constant in every copy, so address arithmetic folds and a loop with
    no taken branches runs in a single simulator cycle. Skipped when the counter is read after
    the loop.
    """

    name = "unroll"

    def run(self, kernel: Kernel) -> None:
        def go(body: list) -> list:
            out: list = []
            for s in body:
                if isinstance(s, For):
                    s.body = go(s.body)
                    trip = s.stop - s.start
                    nested = any(isinstance(t, For) for t in _walk(s.body))
                    if trip <= MAX_UNROLL and not nested and not _uses_outside(kernel, s, s.var):
                        for i in range(trip):
                            for t in copy.deepcopy(s.body):
                                _rewrite_reads(
                                    t, lambda x, v=s.var, c=s.start + i: c if x == v else x
                                )
                                out.append(t)
                        continue
                elif isinstance(s, If):
                    s.body = go(s.body)
                out.append(s)
            return out

        kernel.body = go(kernel.body)


# --- constant folding --------------------------------------------------------------------------


class Fold:
    """Constant propagation and folding on straight-line code (mostly after unrolling).

    * a single-assignment `set t, c` is propagated into its uses;
    * `add`/`mul` of two constants becomes a `set`;
    * constants are moved to the right-hand operand;
    * `add (add x, c1), c2` becomes `add x, c1 + c2` when x cannot change in between
      (a constant, a special register, or a single-assignment vreg);
    * `d = add a, 0` is a copy: reads of d read `a` instead (same condition on `a`).
    Whatever becomes dead is removed by `kernel-dce`.
    """

    name = "fold"

    def run(self, kernel: Kernel) -> None:
        for _ in range(20):  # each round can enable the next; stops at a fixpoint
            if not self._round(kernel):
                break

    def _round(self, kernel: Kernel) -> bool:
        defc = _def_counts(kernel)
        single = {
            _dst(s): s for s in _walk(kernel.body) if _dst(s) is not None and defc[_dst(s)] == 1
        }
        changed = False

        def const_of(x):
            d = single.get(x) if isinstance(x, VReg) else None
            return (
                d.args[0]
                if isinstance(d, Assign) and d.op == "set" and isinstance(d.args[0], int)
                else None
            )

        # `d = add a, 0` is a copy of `a`; reads of d can read a instead (when a cannot change)
        copies = {
            d: st.args[0]
            for d, st in single.items()
            if isinstance(st, Assign)
            and st.op == "add"
            and isinstance(st.args[1], int)
            and st.args[1] == 0
            and not isinstance(st.args[0], int)
            and (not isinstance(st.args[0], VReg) or st.args[0] in single)
        }

        def prop(x):
            nonlocal changed
            while isinstance(x, VReg) and x in copies:  # follow chains of copies
                x = copies[x]
                changed = True
            c = const_of(x)
            if c is not None:
                changed = True
                return c
            return x

        for s in _walk(kernel.body):
            _rewrite_reads_shallow(s, prop)
            if not isinstance(s, Assign) or s.op == "set":
                continue
            a, b = s.args
            if isinstance(a, int) and isinstance(b, int):
                v = _apply(s.op, a, b)
                if I32_LO <= v <= I32_HI:
                    s.op, s.args, changed = "set", (v,), True
                continue
            if isinstance(a, int):  # constants go on the right
                s.args, changed = (b, a), True
                a, b = b, a
            if s.op == "add" and isinstance(b, int) and isinstance(a, VReg):
                inner = single.get(a)
                if (
                    isinstance(inner, Assign)
                    and inner.op == "add"
                    and isinstance(inner.args[1], int)
                    and (not isinstance(inner.args[0], VReg) or inner.args[0] in single)
                ):
                    total = b + inner.args[1]
                    if I32_LO <= total <= I32_HI:
                        s.args, changed = (inner.args[0], total), True
        return changed


def _rewrite_reads_shallow(s, fn: Callable) -> None:
    """Like _rewrite_reads but without recursing (the caller already walks every statement)."""
    if isinstance(s, (For, If)):
        if isinstance(s, If):
            s.a, s.b = fn(s.a), fn(s.b)
        return
    _rewrite_reads(s, fn)


KERNEL_PASSES = {
    p.name: p for p in (LICM, StrengthReduce, KernelDCE, SharedTile, Unroll, Fold)
}  # fmt: skip


def kernel_pass_names(opt_level: int) -> list[str]:
    if opt_level <= 0:
        return []
    if opt_level == 1:
        return ["licm", "strength-reduce", "kernel-dce"]
    return ["licm", "strength-reduce", "shared-tile", "unroll", "fold", "kernel-dce"]
