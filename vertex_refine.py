"""Vertex view-projection refinement.

For each predicted 3D vertex V:

  1. Project V to 2D in every registered COLMAP view.
  2. In each view, find the nearest gestalt corner-class pixel
     (apex / eave_end_point / flashing_end_point) within ``max_pixel_dist``.
  3. If at least ``min_views`` views have a 2D match, DLT-triangulate a new
     3D position from those 2D corner detections.
  4. Sanity-check: the refined position must (a) lie within
     ``max_move_meters`` of the original V, and (b) have mean reprojection
     error below ``max_reproj_px`` across the supporting views.
  5. If both checks pass, replace V with the refined position.

This is pure precision refinement of corner positions. Topology (edges)
is preserved. Falls back to the input on any error — the function is
guaranteed to never return fewer vertices than it received.

Targets corner_f1: the learned model's 3D vertices are approximate; the
gestalt segmentation gives us *exact* pixel-level corner detections per
view; triangulating from those gives a much tighter 3D position whenever
multiple views agree.
"""

from __future__ import annotations

import numpy as np
import cv2


POINT_CLASSES = ("apex", "eave_end_point", "flashing_end_point")


def _build_corner_pixels_per_view(good, views):
    """Build per-view corner-pixel catalog.

    Returns dict ``img_id -> {P, corners (N,2), tree, H, W}``.
    Only includes views that have at least one corner pixel.
    """
    from hoho2025.color_mappings import gestalt_color_mapping
    from scipy.spatial import cKDTree

    out = {}
    for gest_pil, depth_pil, img_id in zip(
        good["gestalt"], good["depth"], good["image_ids"]
    ):
        if img_id not in views:
            continue
        depth_np = np.array(depth_pil)
        H, W = depth_np.shape[:2]
        gest_np = np.array(gest_pil.resize((W, H))).astype(np.uint8)

        corners = []
        for cls in POINT_CLASSES:
            color = np.array(gestalt_color_mapping[cls])
            mask = cv2.inRange(gest_np, color - 0.5, color + 0.5)
            if mask.sum() == 0:
                continue
            n_cc, _, _, centroids = cv2.connectedComponentsWithStats(
                mask, 8, cv2.CV_32S
            )
            if n_cc > 1:
                # Skip the background centroid at index 0; rest are blob centers.
                corners.extend(centroids[1:].tolist())

        if not corners:
            continue
        corners_arr = np.asarray(corners, dtype=np.float64)
        out[img_id] = {
            "P": views[img_id]["P"],
            "corners": corners_arr,
            "tree": cKDTree(corners_arr),
            "H": H,
            "W": W,
        }
    return out


def refine_vertices_view_projection(
    pv,
    pe,
    sample,
    max_pixel_dist: float = 15.0,
    min_views: int = 2,
    max_move_meters: float = 0.5,
    max_reproj_px: float = 10.0,
):
    """Refine predicted vertex positions by re-triangulating from gestalt corners.

    Args:
        pv: (N, 3) predicted vertices in world coordinates.
        pe: edge list (unchanged on return).
        sample: raw dataset entry.
        max_pixel_dist: max 2D distance from projected vertex to a gestalt
            corner pixel to count as a match (15px ~= 2% of 768px width).
        min_views: minimum views with a 2D match to attempt re-triangulation.
        max_move_meters: refined vertex must lie within this distance of
            the original — guards against spurious cross-corner matches.
        max_reproj_px: refined vertex must have mean reprojection error
            below this across the supporting views.

    Returns:
        (pv_refined, pe). Vertex count unchanged; pe is the same list.
        Falls back to (pv, pe) on any error.
    """
    try:
        from hoho2025.example_solutions import convert_entry_to_human_readable
        from mvs_utils import (
            collect_views, project_world_to_image,
            triangulate_dlt, mean_reprojection_error,
        )

        pv_arr = np.asarray(pv, dtype=np.float64)
        if pv_arr.ndim != 2 or pv_arr.shape[0] < 1:
            return pv, pe

        good = convert_entry_to_human_readable(sample)
        colmap_rec = good.get("colmap") or good.get("colmap_binary")
        if colmap_rec is None:
            return pv, pe

        views = collect_views(colmap_rec, good["image_ids"])
        if len(views) < min_views:
            return pv, pe

        view_data = _build_corner_pixels_per_view(good, views)
        if len(view_data) < min_views:
            return pv, pe

        refined = pv_arr.copy()
        n_refined = 0

        for i, v in enumerate(pv_arr):
            Ps_match = []
            pts2d_match = []

            for img_id, vd in view_data.items():
                uv, z = project_world_to_image(vd["P"], v.reshape(1, 3))
                if z[0] <= 0:
                    continue
                u, vp = float(uv[0, 0]), float(uv[0, 1])
                if not (0 <= u < vd["W"] and 0 <= vp < vd["H"]):
                    continue

                dist, idx = vd["tree"].query([u, vp], k=1)
                if dist <= max_pixel_dist:
                    Ps_match.append(vd["P"])
                    pts2d_match.append(vd["corners"][idx])

            if len(Ps_match) < min_views:
                continue  # not enough 2D evidence; keep original V

            v_new = triangulate_dlt(Ps_match, pts2d_match)
            if not np.all(np.isfinite(v_new)):
                continue

            # Sanity 1: refined position shouldn't have moved too far.
            if float(np.linalg.norm(v_new - v)) > max_move_meters:
                continue

            # Sanity 2: refined position must reproject well to its 2D matches.
            err = mean_reprojection_error(v_new, Ps_match, pts2d_match)
            if not np.isfinite(err) or err > max_reproj_px:
                continue

            refined[i] = v_new
            n_refined += 1

        return refined.astype(np.asarray(pv).dtype), pe

    except Exception:
        return pv, pe
