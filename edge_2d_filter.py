"""2D gestalt-edge consistency filter for predicted 3D wireframe edges.

For each predicted edge (v_a, v_b), project the endpoints into every COLMAP view
and walk the 2D segment between them. Count how many sampled pixels fall on a
gestalt "edge class" pixel (eave, ridge, rake, valley, hip, flashing,
step_flashing). If at least ``min_views_support`` views show strong overlap
(>= ``min_pixel_frac`` of samples), the edge is kept; otherwise it is dropped
as a hallucination unsupported by 2D evidence.

All COLMAP cameras + masks are already validated machinery (triangulation
tracks use them). The filter is precision-only — it can drop edges but cannot
introduce new geometry. Conservative defaults: an edge needs evidence in 2+
views, with at least 25% of the projected segment lying on edge-class pixels
(after dilation), to survive.

Falls back to the unfiltered (vertices, edges) on any failure so the pipeline
cannot collapse if a sample's COLMAP / gestalt data is malformed.
"""

from __future__ import annotations

import numpy as np
import cv2

EDGE_CLASSES = (
    "eave", "ridge", "rake", "valley", "hip", "flashing", "step_flashing",
)


def drop_orphan_vertices(pv, pe):
    """Remove vertices that aren't endpoints of any edge and reindex edges.

    A pure precision pass: orphan vertices can only hurt corner_f1 since they
    can't possibly match a ground-truth corner (no edges = nothing to align).
    Side-effect from local A/B (50 samples): when the 2D filter dropped zero
    edges but still helped scores, the gain came entirely from this cleanup
    (e.g. samples 115adff0210, 74844f9fdda, dbec7550263 all 2dfilt=N->N but
    gained +0.04 to +0.06).
    """
    pv_arr = np.asarray(pv)
    if pv_arr.ndim != 2 or pv_arr.shape[0] < 2 or len(pe) < 1:
        return pv, pe
    used = sorted({int(x) for e in pe for x in e if 0 <= int(x) < len(pv_arr)})
    if len(used) < 2:
        return pv, pe
    if len(used) == len(pv_arr):
        return pv, pe  # nothing orphan, fast path
    old_to_new = {old: new for new, old in enumerate(used)}
    new_pv = pv_arr[used]
    new_pe = []
    for a, b in pe:
        a, b = int(a), int(b)
        if a in old_to_new and b in old_to_new:
            new_pe.append((old_to_new[a], old_to_new[b]))
    if len(new_pe) < 1:
        return pv, pe
    return new_pv, new_pe


def _build_edge_masks(good, views, dilate_px: int):
    """Build per-view dilated binary masks of edge-class pixels.

    Returns dict ``img_id -> (mask_bool, H, W)`` for every img_id that has both
    a registered COLMAP view and a gestalt image.
    """
    from hoho2025.color_mappings import gestalt_color_mapping

    out = {}
    for gest_pil, depth_pil, img_id in zip(
        good["gestalt"], good["depth"], good["image_ids"]
    ):
        if img_id not in views:
            continue
        depth_np = np.array(depth_pil)
        H, W = depth_np.shape[:2]
        gest_np = np.array(gest_pil.resize((W, H))).astype(np.uint8)

        mask = np.zeros((H, W), dtype=np.uint8)
        for cls in EDGE_CLASSES:
            color = np.array(gestalt_color_mapping[cls])
            m = cv2.inRange(gest_np, color - 0.5, color + 0.5)
            mask |= m

        if dilate_px > 0:
            k = 2 * dilate_px + 1
            mask = cv2.dilate(mask, np.ones((k, k), np.uint8), iterations=1)

        out[img_id] = (mask > 0, H, W)
    return out


def filter_edges_by_2d_support(
    pv,
    pe,
    sample,
    min_views_support: int = 2,
    min_pixel_frac: float = 0.25,
    dilate_px: int = 4,
    sample_steps: int = 20,
):
    """Drop edges whose 2D projection lacks gestalt-edge mask support.

    Args:
        pv: (N, 3) vertices in world coordinates.
        pe: list of (u, v) edge indices.
        sample: raw dataset entry (used to access COLMAP + gestalt views).
        min_views_support: edge must be supported by this many views to be kept.
        min_pixel_frac: fraction of sampled pixels along the projected line
            that must lie on an edge-class pixel for the view to count as
            supporting.
        dilate_px: pixel-radius dilation of the edge mask (gives tolerance
            for slightly-off projections; 4 → 9px-thick edges).
        sample_steps: number of points sampled along each 2D projected line.

    Returns:
        (pv_filtered, pe_filtered) with orphaned vertices removed. Falls back
        to the input on any error or if the filter would leave fewer than one
        edge or two vertices.
    """
    try:
        from hoho2025.example_solutions import convert_entry_to_human_readable
        from mvs_utils import collect_views, project_world_to_image

        pv_arr = np.asarray(pv, dtype=np.float64)
        if pv_arr.ndim != 2 or pv_arr.shape[0] < 2 or len(pe) < 1:
            return pv, pe

        good = convert_entry_to_human_readable(sample)
        colmap_rec = good.get("colmap") or good.get("colmap_binary")
        if colmap_rec is None:
            return pv, pe

        views = collect_views(colmap_rec, good["image_ids"])
        if len(views) < min_views_support:
            return pv, pe

        view_masks = _build_edge_masks(good, views, dilate_px=dilate_px)
        if not view_masks:
            return pv, pe

        keep_edges = []
        for u, v in pe:
            u, v = int(u), int(v)
            if u == v or u >= len(pv_arr) or v >= len(pv_arr):
                continue
            endpoints = np.stack([pv_arr[u], pv_arr[v]])

            supporting = 0
            for img_id, view in views.items():
                if img_id not in view_masks:
                    continue
                mask_bool, H, W = view_masks[img_id]

                uv, z = project_world_to_image(view["P"], endpoints)
                # Require both endpoints in front of camera and inside the image.
                if z[0] <= 0 or z[1] <= 0:
                    continue
                if not (
                    0 <= uv[0, 0] < W
                    and 0 <= uv[0, 1] < H
                    and 0 <= uv[1, 0] < W
                    and 0 <= uv[1, 1] < H
                ):
                    continue

                t = np.linspace(0.0, 1.0, sample_steps)
                xs = uv[0, 0] + t * (uv[1, 0] - uv[0, 0])
                ys = uv[0, 1] + t * (uv[1, 1] - uv[0, 1])
                xs_i = np.clip(xs.astype(np.int32), 0, W - 1)
                ys_i = np.clip(ys.astype(np.int32), 0, H - 1)

                hits = int(mask_bool[ys_i, xs_i].sum())
                if hits / float(sample_steps) >= min_pixel_frac:
                    supporting += 1
                    if supporting >= min_views_support:
                        break

            if supporting >= min_views_support:
                keep_edges.append((u, v))

        # Safety: never return an empty graph.
        if len(keep_edges) < 1:
            return pv, pe

        used = sorted({a for e in keep_edges for a in e})
        if len(used) < 2:
            return pv, pe
        old_to_new = {old: new for new, old in enumerate(used)}
        new_pv = np.asarray([pv_arr[old] for old in used])
        new_pe = [(old_to_new[a], old_to_new[b]) for a, b in keep_edges]
        return new_pv, new_pe

    except Exception:
        return pv, pe


def filter_edges_strict_no_support(
    pv,
    pe,
    sample,
    max_support_thresh: float = 0.10,
    dilate_px: int = 4,
    sample_steps: int = 20,
):
    """Drop only edges that have NO 2D support in any view (clear hallucinations).

    Asymmetric to filter_edges_by_2d_support: instead of requiring N views to
    support an edge, we only drop an edge if its MAX support across all views
    is below max_support_thresh. So an edge supported by even ONE view at 30%
    is kept; only edges with universally weak overlap (max < 10%) are dropped.

    Designed for the case where the symmetric filter (50e27b8) over-pruned the
    q5 worst-cases (local A/B showed q5: 0.123 → 0.077 with that filter).
    """
    try:
        from hoho2025.example_solutions import convert_entry_to_human_readable
        from mvs_utils import collect_views, project_world_to_image

        pv_arr = np.asarray(pv, dtype=np.float64)
        if pv_arr.ndim != 2 or pv_arr.shape[0] < 2 or len(pe) < 1:
            return pv, pe

        good = convert_entry_to_human_readable(sample)
        colmap_rec = good.get("colmap") or good.get("colmap_binary")
        if colmap_rec is None:
            return pv, pe

        views = collect_views(colmap_rec, good["image_ids"])
        if len(views) < 1:
            return pv, pe

        view_masks = _build_edge_masks(good, views, dilate_px=dilate_px)
        if not view_masks:
            return pv, pe

        keep_edges = []
        for u, v in pe:
            u, v = int(u), int(v)
            if u == v or u >= len(pv_arr) or v >= len(pv_arr):
                continue
            endpoints = np.stack([pv_arr[u], pv_arr[v]])

            max_support = 0.0
            for img_id, view in views.items():
                if img_id not in view_masks:
                    continue
                mask_bool, H, W = view_masks[img_id]

                uv, z = project_world_to_image(view["P"], endpoints)
                if z[0] <= 0 or z[1] <= 0:
                    continue
                if not (
                    0 <= uv[0, 0] < W and 0 <= uv[0, 1] < H
                    and 0 <= uv[1, 0] < W and 0 <= uv[1, 1] < H
                ):
                    continue

                t = np.linspace(0.0, 1.0, sample_steps)
                xs = uv[0, 0] + t * (uv[1, 0] - uv[0, 0])
                ys = uv[0, 1] + t * (uv[1, 1] - uv[0, 1])
                xs_i = np.clip(xs.astype(np.int32), 0, W - 1)
                ys_i = np.clip(ys.astype(np.int32), 0, H - 1)

                support = int(mask_bool[ys_i, xs_i].sum()) / float(sample_steps)
                if support > max_support:
                    max_support = support
                if max_support >= max_support_thresh:
                    break  # early exit, edge is safe

            if max_support >= max_support_thresh:
                keep_edges.append((u, v))

        if len(keep_edges) < 1:
            return pv, pe

        # Orphan vertex cleanup (the part that actually helps in A/B testing).
        used = sorted({a for e in keep_edges for a in e})
        if len(used) < 2:
            return pv, pe
        old_to_new = {old: new for new, old in enumerate(used)}
        new_pv = np.asarray([pv_arr[old] for old in used])
        new_pe = [(old_to_new[a], old_to_new[b]) for a, b in keep_edges]
        return new_pv, new_pe

    except Exception:
        return pv, pe
