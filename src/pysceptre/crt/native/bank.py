import cupy as cp
import numpy as np

from .stream import gpu_exp


def bank(p, B, rng):
    p = np.clip(np.asarray(p, float), 0, 1)
    if np.any(~np.isfinite(p)) or np.any(p >= 1 / 3):
        raise ValueError("CPU replay required for geometric search branch")
    live = np.flatnonzero(p > 0)
    cursor = np.full(len(live), -1, np.int64)
    cells = []
    draws = []
    passes = 0

    def walk(todo):
        pj = p[live[todo]]
        mean = B * pj
        ng = np.ceil(mean + 6 * np.sqrt(mean * (1 - pj)) + 10).astype(np.int64)
        m = int(ng.sum())
        dg = cp.asarray(ng)
        dt = cp.asarray(todo)
        dl = cp.asarray(live)
        ex, _ = gpu_exp(rng, m)
        ends = cp.asarray(np.cumsum(ng))
        starts = ends - dg
        local = cp.searchsorted(ends, cp.arange(m, dtype=cp.int64), side="right")
        denom = cp.asarray(np.log1p(-pj))[local]
        q = -ex / denom
        # Guard integer-boundary ambiguity before changing the public sampler.
        near = cp.abs(q - cp.rint(q)) <= 64 * np.finfo(float).eps * cp.maximum(1, cp.abs(q))
        if bool(cp.any(near).get()):
            raise RuntimeError("Integer-boundary replay required")
        gaps = cp.minimum(cp.ceil(q), B + 1).astype(cp.int64)
        steps = cp.cumsum(gaps)
        steps -= (steps[starts] - gaps[starts])[local]
        pos = cp.asarray(cursor[todo])[local] + steps
        owner = dt[local]
        keep = pos < B
        cells.append(dl[owner[keep]])
        draws.append(pos[keep])
        cursor[todo] = pos[ends - 1].get()

    first = np.ceil(B * p[live] + 6 * np.sqrt(B * p[live] * (1 - p[live])) + 10)
    bounds = np.searchsorted(
        np.cumsum(first), np.arange(1, 1 + int(first.sum() // 4000000)) * 4000000
    )
    for block in np.split(np.arange(live.size), bounds):
        if block.size:
            walk(block)
    todo = np.flatnonzero(cursor < B)
    while todo.size:
        walk(todo)
        todo = todo[cursor[todo] < B]
        passes += 1
    c = cp.concatenate(cells) if cells else cp.empty(0, cp.int64)
    d = cp.concatenate(draws) if draws else cp.empty(0, cp.int64)
    # Unique (draw,cell) keys preserve ascending cell order exactly.
    key = d * np.int64(len(p)) + c
    key.sort()
    c = (key % len(p)).astype(cp.int32)
    d = key // len(p)
    ptr = cp.empty(B + 1, cp.int64)
    ptr[0] = 0
    ptr[1:] = cp.cumsum(cp.bincount(d, minlength=B))
    return c, ptr
