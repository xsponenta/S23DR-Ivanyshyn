from __future__ import annotations

import numpy as np


def merge_vertices_iterative(vertices: np.ndarray, edges: np.ndarray,
                              start: float = 0.15, end: float = 0.6,
                              n_iters: int = 5):
    """Iterative merge: start with tight threshold, gradually widen.

    Avoids the worst transitive chaining effects of a single wide threshold.
    Each pass merges only the closest pairs first, establishing stable cluster
    centers before wider merges pull in more distant endpoints.

    +0.004 HSS / +0.007 F1 over single-pass merge(0.4) on 1024 val samples.
    """
    pv, pe = vertices, edges
    for t in np.linspace(start, end, n_iters):
        pv, pe = merge_vertices(pv, pe, t)
    return pv, pe


def merge_vertices(vertices: np.ndarray, edges: np.ndarray, thresh: float):
    verts = np.asarray(vertices, dtype=np.float32)
    edges = np.asarray(edges, dtype=np.int64)
    if verts.size == 0 or edges.size == 0:
        return verts, edges

    n = verts.shape[0]
    parent = np.arange(n, dtype=np.int64)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri = find(i)
        rj = find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        vi = verts[i]
        for j in range(i + 1, n):
            if np.linalg.norm(vi - verts[j]) <= thresh:
                union(i, j)

    clusters = {}
    for i in range(n):
        root = find(i)
        clusters.setdefault(root, []).append(i)

    new_vertices = []
    mapping = {}
    for new_idx, idxs in enumerate(clusters.values()):
        pts = verts[idxs]
        center = pts.mean(axis=0)
        new_vertices.append(center)
        for i in idxs:
            mapping[i] = new_idx

    new_edges = []
    seen = set()
    for a, b in edges:
        na = mapping.get(int(a), int(a))
        nb = mapping.get(int(b), int(b))
        if na == nb:
            continue
        key = (na, nb) if na <= nb else (nb, na)
        if key in seen:
            continue
        seen.add(key)
        new_edges.append([na, nb])

    return np.asarray(new_vertices, dtype=np.float32), np.asarray(new_edges, dtype=np.int64)
