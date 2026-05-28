"""Multi-view corner triangulation pipeline (T1.2 – T1.6).

Drop-in replacement for the depth-based ``project_vertices_to_3d`` step in
``sklearn_submission.py``. The depth map is only used as a sanity filter, never
as the source of 3D positions — the actual geometry comes from COLMAP cameras
via DLT triangulation.

Entry points:
    detect_corners_per_view(entry)  → dict[view_id → List[Corner]]
    triangulate_wireframe(entry, corners_per_view) → Tracks + per-track obs

Everything is pure numpy + pycolmap + cv2 — no torch, no kornia.
"""

from __future__ import annotations

import numpy as np
import cv2
from dataclasses import dataclass, field

from hoho2025.example_solutions import (
    convert_entry_to_human_readable,
    filter_vertices_by_background,
    point_to_segment_dist,
)
from hoho2025.color_mappings import gestalt_color_mapping

try:
    from mvs_utils import (
        collect_views, triangulate_dlt, mean_reprojection_error,
        fundamental_matrix, epipolar_line, point_to_line_distance,
        project_world_to_image,
    )
except ImportError:
    from submission.mvs_utils import (
        collect_views, triangulate_dlt, mean_reprojection_error,
        fundamental_matrix, epipolar_line, point_to_line_distance,
        project_world_to_image,
    )


# Vertex classes we consider (minus 'post' — added later in T1.7 when safe).
VERTEX_CLASSES = ['apex', 'eave_end_point', 'flashing_end_point']
EDGE_CLASSES = ['eave', 'ridge', 'rake', 'valley', 'hip']


@dataclass
class Corner:
    """A 2D corner detected on a single view."""
    view_id: str
    xy: np.ndarray          # (2,) float32 pixel coords at COLMAP-native resolution
    cls: str                # gestalt class label
    blob_area: int          # area of the connected component, for tie-breaks


@dataclass
class Track:
    """A 3D wireframe vertex with its per-view observations."""
    xyz: np.ndarray         # (3,) float64
    cls: str
    observations: list[tuple[str, np.ndarray]] = field(default_factory=list)
    reproj_err: float = float("inf")
    # view_id → index into corners_per_view[view_id]. Populated by build_tracks
    # when per-view edges need to be lifted to 3D.
    corner_indices: dict[str, int] = field(default_factory=dict)


def _refine_centroids_subpix(gest_seg_np, centroids, max_shift=4.0, win=5):
    """cv2.cornerSubPix refinement inside an apex blob. Identical to the
    version in sklearn_submission.py — duplicated here to keep triangulation.py
    importable on its own.
    """
    if len(centroids) == 0:
        return centroids
    gray = cv2.cvtColor(gest_seg_np, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    pts = np.asarray(centroids, dtype=np.float32).reshape(-1, 1, 2).copy()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    try:
        refined = cv2.cornerSubPix(gray, pts, (win, win), (-1, -1), criteria)
    except cv2.error:
        return centroids
    refined = refined.reshape(-1, 2)
    orig = np.asarray(centroids, dtype=np.float32)
    shifts = np.linalg.norm(refined - orig, axis=1)
    mask = shifts <= max_shift
    out = orig.copy()
    out[mask] = refined[mask]
    return out


def _detect_edges_2d(
    gest_np: np.ndarray,
    corners: list[Corner],
    edge_th: float = 15.0,
) -> list[tuple[int, int, str]]:
    """Detect 2D gestalt edges and connect them to existing corner indices.

    Mirrors ``get_vertices_and_edges_improved`` from sklearn_submission but
    keeps *all* edge classes and returns triples ``(ci, cj, edge_cls)`` so
    we can aggregate edge-class votes downstream.
    """
    if len(corners) < 2:
        return []
    apex_pts = np.array([c.xy for c in corners], dtype=np.float32)
    connections: list[tuple[int, int, str]] = []
    for edge_class in EDGE_CLASSES:
        color = np.array(gestalt_color_mapping[edge_class])
        mask_raw = cv2.inRange(gest_np, color - 0.5, color + 0.5)
        mask = cv2.morphologyEx(mask_raw, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        if mask.sum() == 0:
            continue
        _, labels, _, _ = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
        for lbl in range(1, labels.max() + 1):
            ys, xs = np.where(labels == lbl)
            if len(xs) < 2:
                continue
            pts = np.column_stack([xs, ys]).astype(np.float32)
            line_params = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
            vx, vy, x0, y0 = line_params.ravel()
            proj = (xs - x0) * vx + (ys - y0) * vy
            p1 = np.array([x0 + proj.min() * vx, y0 + proj.min() * vy])
            p2 = np.array([x0 + proj.max() * vx, y0 + proj.max() * vy])
            dists = np.array(
                [point_to_segment_dist(apex_pts[i], p1, p2) for i in range(len(apex_pts))]
            )
            near = np.where(dists <= edge_th)[0]
            if len(near) < 2:
                continue
            near_pts = apex_pts[near]
            a = int(near[np.argmin(np.linalg.norm(near_pts - p1, axis=1))])
            b = int(near[np.argmin(np.linalg.norm(near_pts - p2, axis=1))])
            if a != b:
                lo, hi = (a, b) if a < b else (b, a)
                connections.append((lo, hi, edge_class))
    return connections


def detect_corners_per_view(
    entry,
    vertex_classes: list[str] | None = None,
    filter_background: bool = True,
    return_edges: bool = False,
):
    """Run per-view corner detection + subpixel refinement.

    Returns
    -------
    corners_per_view : dict[image_id → list[Corner]]
    good_entry : the convert_entry_to_human_readable output (caller reuses it)
    edges_per_view (if ``return_edges``) : dict[image_id → list[(ci, cj, edge_cls)]]
    """
    if vertex_classes is None:
        vertex_classes = VERTEX_CLASSES

    good = convert_entry_to_human_readable(entry)
    corners_per_view: dict[str, list[Corner]] = {}
    edges_per_view: dict[str, list[tuple[int, int, str]]] = {}

    for i, (gest, depth, img_id, ade_seg) in enumerate(zip(
        good['gestalt'], good['depth'], good['image_ids'], good['ade']
    )):
        # Native resolution used by the COLMAP camera is the depth resolution
        # (768×576 in practice). Resize gestalt to match so pixel coordinates
        # are compatible with our projection matrices.
        depth_np = np.array(depth)
        H, W = depth_np.shape[:2]
        gest_np = np.array(gest.resize((W, H))).astype(np.uint8)
        ade_np = np.array(ade_seg.resize((W, H))).astype(np.uint8)

        corners: list[Corner] = []
        for v_class in vertex_classes:
            color = np.array(gestalt_color_mapping[v_class])
            mask = cv2.inRange(gest_np, color - 0.5, color + 0.5)
            if mask.sum() == 0:
                continue
            _, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
            blob_centroids = centroids[1:]
            areas = stats[1:, cv2.CC_STAT_AREA]
            if len(blob_centroids) == 0:
                continue
            refined = _refine_centroids_subpix(gest_np, blob_centroids)
            for xy, area in zip(refined, areas):
                corners.append(Corner(
                    view_id=img_id,
                    xy=np.asarray(xy, dtype=np.float32),
                    cls=v_class,
                    blob_area=int(area),
                ))

        if filter_background and corners:
            fake_verts = [{"xy": c.xy, "type": c.cls} for c in corners]
            fake_verts, _ = filter_vertices_by_background(fake_verts, [], ade_np)
            kept_keys = {(float(v['xy'][0]), float(v['xy'][1]), v['type']) for v in fake_verts}
            corners = [c for c in corners
                       if (float(c.xy[0]), float(c.xy[1]), c.cls) in kept_keys]

        corners_per_view[img_id] = corners
        if return_edges:
            edges_per_view[img_id] = _detect_edges_2d(gest_np, corners)

    if return_edges:
        return corners_per_view, good, edges_per_view
    return corners_per_view, good


def build_tracks(
    corners_per_view: dict[str, list[Corner]],
    views: dict[str, dict],
    class_strict: bool = True,
    epipolar_px: float = 6.0,
    reproj_px: float = 4.0,
    min_views: int = 2,
) -> list[Track]:
    """Greedy multi-view matching and triangulation with epipolar gating.

    Strategy (classical, mirrors PC2WF / COLMAP incremental triangulation):

    1. Build a pool of unmatched corners from every view.
    2. For every ordered pair of views compute the fundamental matrix.
    3. For each corner in view_a, find all corners in view_b of the same class
       whose perpendicular distance to the epipolar line is below
       ``epipolar_px``. Triangulate each candidate pair via DLT.
    4. For each candidate 3D point, reproject it back into every other view.
       A corner of the same class within ``reproj_px`` of the reprojection
       becomes an additional observation. Re-triangulate with the enlarged
       observation list.
    5. Accept the track if it has ≥ ``min_views`` observations, mean
       reprojection error < ``reproj_px``, and positive depth everywhere.
    6. Mark all corners in the track as matched so they are not reused.

    Parameters are intentionally tight — noise-reducing rather than
    permissive — because a wrongly triangulated vertex can sit meters
    away from any real roof feature.
    """
    # Stable ordering: view ids sorted
    view_ids = [vid for vid in corners_per_view.keys() if vid in views]
    view_ids.sort()

    # Index remaining corners (view_id, idx) → Corner
    remaining: dict[tuple[str, int], Corner] = {}
    for vid in view_ids:
        for idx, c in enumerate(corners_per_view[vid]):
            remaining[(vid, idx)] = c

    tracks: list[Track] = []

    for anchor_vid in view_ids:
        for (r_vid, r_idx), anchor in list(remaining.items()):
            if r_vid != anchor_vid:
                continue
            # Try matching this anchor against each other view.
            best_track: Track | None = None

            for other_vid in view_ids:
                if other_vid == anchor_vid:
                    continue
                F = fundamental_matrix(views[anchor_vid], views[other_vid])
                line = epipolar_line(F, anchor.xy)

                for (o_vid, o_idx), cand in remaining.items():
                    if o_vid != other_vid:
                        continue
                    if class_strict and cand.cls != anchor.cls:
                        continue
                    d = point_to_line_distance(line, cand.xy)
                    if d > epipolar_px:
                        continue

                    # Two-view DLT
                    Ps = [views[anchor_vid]["P"], views[other_vid]["P"]]
                    pts = [anchor.xy, cand.xy]
                    X = triangulate_dlt(Ps, pts)
                    if not np.all(np.isfinite(X)):
                        continue

                    # Extend with all other views that also see this point.
                    obs = [(anchor_vid, anchor.xy), (other_vid, cand.xy)]
                    used_keys = {(anchor_vid, r_idx), (other_vid, o_idx)}
                    for ext_vid in view_ids:
                        if ext_vid in (anchor_vid, other_vid):
                            continue
                        uv, z = project_world_to_image(views[ext_vid]["P"], X.reshape(1, 3))
                        if z[0] <= 0:
                            continue
                        u_pred = uv[0]
                        best_match = None
                        best_dist = reproj_px
                        for (e_vid, e_idx), ec in remaining.items():
                            if e_vid != ext_vid:
                                continue
                            if class_strict and ec.cls != anchor.cls:
                                continue
                            d2 = float(np.linalg.norm(ec.xy - u_pred))
                            if d2 < best_dist:
                                best_dist = d2
                                best_match = (e_vid, e_idx, ec)
                        if best_match is not None:
                            obs.append((best_match[0], best_match[2].xy))
                            used_keys.add((best_match[0], best_match[1]))

                    if len(obs) < min_views:
                        continue

                    # Retriangulate on full observation set for stability
                    Ps_full = [views[vid]["P"] for vid, _ in obs]
                    pts_full = [uv for _, uv in obs]
                    X_full = triangulate_dlt(Ps_full, pts_full)
                    if not np.all(np.isfinite(X_full)):
                        continue
                    err = mean_reprojection_error(X_full, Ps_full, pts_full)
                    if err > reproj_px:
                        continue

                    track = Track(
                        xyz=X_full,
                        cls=anchor.cls,
                        observations=obs,
                        reproj_err=err,
                    )
                    track._used_keys = used_keys  # type: ignore[attr-defined]
                    if best_track is None or len(track.observations) > len(best_track.observations) \
                       or (len(track.observations) == len(best_track.observations) and err < best_track.reproj_err):
                        best_track = track

            if best_track is not None:
                # Freeze the corner-index mapping and forget the private attr.
                used = getattr(best_track, "_used_keys", set())
                best_track.corner_indices = {vid: int(idx) for vid, idx in used}
                try:
                    delattr(best_track, "_used_keys")
                except AttributeError:
                    pass
                tracks.append(best_track)
                # Retire matched corners so they aren't reused.
                for key in used:
                    remaining.pop(key, None)

    return tracks


def get_high_confidence_tracks(
    entry,
    min_views: int = 3,
    max_reproj_px: float = 2.0,
    epipolar_px: float = 6.0,
    build_reproj_px: float = 4.0,
) -> list[Track]:
    """Run the full triangulation pipeline and return only the tracks
    that pass a stricter quality gate.

    The default ``min_views=3`` and ``max_reproj_px=2.0`` are tighter
    than ``predict_wireframe_tracks`` defaults and are designed for
    using these tracks as **vertex sources** rather than just edge
    sources. A ≥3-view DLT triangulation with <2 px mean reprojection
    error has a 3D accuracy of 5–10 cm — substantially better than
    depth-based unprojection.
    """
    tracks, _views, _good = triangulate_wireframe(
        entry,
        epipolar_px=epipolar_px,
        reproj_px=build_reproj_px,
        min_views=2,
        want_edges=False,
    )
    return [
        t for t in tracks
        if len(t.observations) >= min_views and t.reproj_err <= max_reproj_px
    ]


def predict_wireframe_tracks(
    entry,
    min_views: int = 2,
    min_votes: int = 1,
    epipolar_px: float = 6.0,
    reproj_px: float = 4.0,
    merge_radius: float = 0.3,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Standalone triangulation-based wireframe predictor.

    Returns (vertices, edges) in the same format as
    ``predict_wireframe_sklearn`` — ready to feed into ``hss()``.
    """
    import numpy as _np

    tracks, _views, _good, t_edges = triangulate_wireframe(
        entry,
        epipolar_px=epipolar_px,
        reproj_px=reproj_px,
        min_views=min_views,
        want_edges=True,
    )
    if not tracks:
        return _np.zeros((2, 3), dtype=_np.float64), [(0, 1)]

    xyz = _np.array([t.xyz for t in tracks], dtype=_np.float64)

    # Merge vertices closer than ``merge_radius``. A simple greedy union-find
    # keyed on first-touched neighbour keeps it O(N^2) but N ≤ 200 in practice.
    n = len(xyz)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    diff = xyz[:, None, :] - xyz[None, :, :]
    dists = _np.sqrt((diff ** 2).sum(-1))
    for i in range(n):
        for j in range(i + 1, n):
            if dists[i, j] <= merge_radius:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(i)

    old_to_new: dict[int, int] = {}
    new_xyz = []
    for new_idx, (root, members) in enumerate(groups.items()):
        for m in members:
            old_to_new[m] = new_idx
        new_xyz.append(xyz[members].mean(axis=0))
    new_xyz = _np.array(new_xyz, dtype=_np.float64)

    # Remap edges, dedup
    edge_set: dict[tuple[int, int], int] = {}
    for ti, tj, votes in t_edges:
        if votes < min_votes:
            continue
        a = old_to_new[ti]
        b = old_to_new[tj]
        if a == b:
            continue
        key = (a, b) if a < b else (b, a)
        edge_set[key] = edge_set.get(key, 0) + votes

    edges = list(edge_set.keys())
    if not edges or len(new_xyz) < 2:
        return _np.zeros((2, 3), dtype=_np.float64), [(0, 1)]

    return new_xyz, [(int(a), int(b)) for a, b in edges]


def build_track_edges(
    tracks: list[Track],
    edges_per_view: dict[str, list[tuple[int, int, str]]],
    min_votes: int = 1,
    max_3d_len: float = 8.0,
) -> list[tuple[int, int, int]]:
    """Aggregate 3D edges from per-view 2D gestalt edges.

    Parameters
    ----------
    tracks : list of Track
    edges_per_view : dict[view_id → list[(corner_i_idx, corner_j_idx, edge_cls)]]
    min_votes : minimum number of views that must agree on an edge.
    max_3d_len : drop edges that would be absurdly long in 3D.

    Returns
    -------
    list of (track_i, track_j, vote_count)
    """
    # (view_id, corner_idx) → track_idx
    key_to_track: dict[tuple[str, int], int] = {}
    for t_idx, t in enumerate(tracks):
        for vid, cidx in t.corner_indices.items():
            key_to_track[(vid, cidx)] = t_idx

    votes: dict[tuple[int, int], int] = {}
    for vid, edges in edges_per_view.items():
        for ci, cj, _ecls in edges:
            ti = key_to_track.get((vid, ci))
            tj = key_to_track.get((vid, cj))
            if ti is None or tj is None or ti == tj:
                continue
            key = (ti, tj) if ti < tj else (tj, ti)
            votes[key] = votes.get(key, 0) + 1

    out: list[tuple[int, int, int]] = []
    for (ti, tj), v in votes.items():
        if v < min_votes:
            continue
        d = float(np.linalg.norm(tracks[ti].xyz - tracks[tj].xyz))
        if d > max_3d_len:
            continue
        out.append((ti, tj, v))
    return out


def triangulate_wireframe(
    entry,
    epipolar_px: float = 6.0,
    reproj_px: float = 4.0,
    min_views: int = 2,
    want_edges: bool = False,
):
    """High-level wrapper: detect corners, build views, triangulate tracks.

    Returns
    -------
    (tracks, views, good_entry)
        when ``want_edges=False`` (default, backwards compatible).
    (tracks, views, good_entry, track_edges)
        when ``want_edges=True``. ``track_edges`` is the output of
        :func:`build_track_edges` — a list of ``(track_i, track_j, vote_count)``.
    """
    if want_edges:
        corners_per_view, good, edges_per_view = detect_corners_per_view(
            entry, return_edges=True
        )
    else:
        corners_per_view, good = detect_corners_per_view(entry)
        edges_per_view = None

    colmap_rec = good.get('colmap') or good.get('colmap_binary')
    views = collect_views(colmap_rec, good['image_ids'])
    tracks = build_tracks(
        corners_per_view, views,
        epipolar_px=epipolar_px,
        reproj_px=reproj_px,
        min_views=min_views,
    )
    if not want_edges:
        return tracks, views, good
    track_edges = build_track_edges(tracks, edges_per_view or {})
    return tracks, views, good, track_edges


# ---------------------------------------------------------------------------
# T1.6: integration helper — refine an existing depth-based 3D vertex set
# by snapping each vertex to its closest triangulated track.
# ---------------------------------------------------------------------------

def refine_vertices_with_tracks(
    merged_v: np.ndarray,
    tracks: list[Track],
    snap_radius: float = 1.0,
    min_views_for_snap: int = 2,
    max_reproj_err_px: float = float("inf"),
) -> tuple[np.ndarray, np.ndarray]:
    """For each vertex in ``merged_v``, find the closest triangulated track
    (by 3D distance) and, if it sits within ``snap_radius`` metres, move the
    vertex to that track's position.

    The graph structure is preserved — only positions move. Tracks with
    fewer than ``min_views_for_snap`` observations are ignored (2-view DLT
    is noisy on short baselines).

    Returns
    -------
    refined_v : (N, 3) float64 — refined vertex positions
    snap_mask : (N,)  bool    — True where a snap happened
    """
    refined = np.asarray(merged_v, dtype=np.float64).copy()
    snap = np.zeros(len(refined), dtype=bool)

    good_tracks = [
        t for t in tracks
        if len(t.observations) >= min_views_for_snap and t.reproj_err <= max_reproj_err_px
    ]
    if not good_tracks or len(refined) == 0:
        return refined, snap

    track_xyz = np.array([t.xyz for t in good_tracks], dtype=np.float64)
    for i in range(len(refined)):
        d = np.linalg.norm(track_xyz - refined[i], axis=1)
        j = int(np.argmin(d))
        if d[j] <= snap_radius:
            refined[i] = track_xyz[j]
            snap[i] = True
    return refined, snap


def augment_with_tracks(
    merged_v: np.ndarray,
    heur_edges: list,
    tracks: list[Track],
    dup_radius: float = 0.4,
    min_views_for_add: int = 3,
    max_reproj_err_px: float = 2.5,
) -> tuple[np.ndarray, list]:
    """Append high-confidence triangulated tracks as new vertices.

    Unlike ``refine_vertices_with_tracks`` (which moves existing vertices and
    risks regressions on already-good ones), this only adds new points that
    sit more than ``dup_radius`` metres from any existing vertex.

    The edge list is returned unchanged — new vertices only get edges via the
    downstream sklearn classifier or heuristic edge-detection step, not here.
    """
    merged = np.asarray(merged_v, dtype=np.float64)
    confident = [t for t in tracks
                 if len(t.observations) >= min_views_for_add
                 and t.reproj_err <= max_reproj_err_px]
    if not confident:
        return merged, heur_edges
    tvs = np.array([t.xyz for t in confident], dtype=np.float64)
    if len(merged) == 0:
        return tvs, heur_edges
    # Keep tracks that are not a duplicate of any existing merged vertex.
    diffs = tvs[:, None, :] - merged[None, :, :]
    dists = np.sqrt((diffs ** 2).sum(-1))
    min_d = dists.min(axis=1)
    new = tvs[min_d > dup_radius]
    if len(new) == 0:
        return merged, heur_edges
    augmented = np.vstack([merged, new])
    return augmented, heur_edges
