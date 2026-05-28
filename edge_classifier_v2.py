"""V2 edge classifier with image-feature inputs.

Richer features than v1: per-edge-class support along projection, endpoint
distance to gestalt corner pixels, view-consistency stats. The signal is what
the gestalt segmentation actually shows along the projected edge — not just
"are there any edge pixels", but WHICH class and HOW consistently.

Feature layout (40 total):
    0-11   geometric features (same as v1)
   12-18   per-edge-class mean fraction along projected segment (7 classes)
   19-25   per-edge-class MAX fraction (best single view)
   26      endpoint A: mean px distance to nearest gestalt corner (clipped 30)
   27      endpoint B: same
   28      endpoint A: count of views where it's <10px from a corner
   29      endpoint B: same
   30      edge midpoint: mean px dist to nearest gestalt edge pixel
   31      num views with any-edge support > 0.3
   32      num views with any-edge support > 0.6
   33      mean projected-length in pixels (across views)
   34      std of any-edge support across views (consistency)
   35      mean depth z (camera frame, across views) — flat-roof prior
   36      colmap support: min endpoint dist to colmap point
   37      colmap support: midpoint dist to colmap point
   38      edge length / scene median length (raw ratio)
   39      vertex_count / scene_median_count (graph context)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

EDGE_CLASSES = (
    "eave", "ridge", "rake", "valley", "hip", "flashing", "step_flashing",
)
POINT_CLASSES = ("apex", "eave_end_point", "flashing_end_point")
NUM_FEATURES = 40


class EdgeClassifierV2(nn.Module):
    def __init__(self, in_dim: int = NUM_FEATURES, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _build_class_masks_and_corner_dt(good, views, dilate_px: int = 3):
    """Per-view: dict img_id -> {edge_masks[7], corner_dt, H, W}."""
    import cv2
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

        # Edge class masks: each cls -> bool mask
        edge_masks = []
        for cls in EDGE_CLASSES:
            color = np.array(gestalt_color_mapping[cls])
            m = cv2.inRange(gest_np, color - 0.5, color + 0.5)
            if dilate_px > 0:
                k = 2 * dilate_px + 1
                m = cv2.dilate(m, np.ones((k, k), np.uint8), iterations=1)
            edge_masks.append(m > 0)

        # Corner DT: distance to nearest pixel of ANY point class
        corner_mask = np.zeros((H, W), dtype=np.uint8)
        for cls in POINT_CLASSES:
            color = np.array(gestalt_color_mapping[cls])
            corner_mask |= cv2.inRange(gest_np, color - 0.5, color + 0.5)
        if corner_mask.sum() > 0:
            corner_dt = cv2.distanceTransform(255 - corner_mask, cv2.DIST_L2, 5)
            corner_dt = np.minimum(corner_dt, 30.0).astype(np.float32)
        else:
            corner_dt = np.full((H, W), 30.0, dtype=np.float32)

        # Edge-union DT (for midpoint feature)
        edge_union = np.zeros((H, W), dtype=np.uint8)
        for cls in EDGE_CLASSES:
            color = np.array(gestalt_color_mapping[cls])
            edge_union |= cv2.inRange(gest_np, color - 0.5, color + 0.5)
        if edge_union.sum() > 0:
            edge_dt = cv2.distanceTransform(255 - edge_union, cv2.DIST_L2, 5)
            edge_dt = np.minimum(edge_dt, 30.0).astype(np.float32)
        else:
            edge_dt = np.full((H, W), 30.0, dtype=np.float32)

        out[img_id] = {
            "edge_masks": edge_masks,  # list of 7 bool (H,W)
            "corner_dt": corner_dt,
            "edge_dt": edge_dt,
            "H": H, "W": W,
        }
    return out


def _edge_features_v2(pv, edges, sample, sample_steps: int = 24):
    """Per-edge feature vectors (E, 40). All zeros on failure."""
    from hoho2025.example_solutions import convert_entry_to_human_readable
    from mvs_utils import collect_views, project_world_to_image
    from scipy.spatial import cKDTree

    E = len(edges)
    feats = np.zeros((E, NUM_FEATURES), dtype=np.float32)
    if E == 0:
        return feats

    pv_arr = np.asarray(pv, dtype=np.float64)

    # Vertex degrees
    deg = np.zeros(len(pv_arr), dtype=np.int32)
    for a, b in edges:
        if 0 <= a < len(pv_arr): deg[a] += 1
        if 0 <= b < len(pv_arr): deg[b] += 1

    # Edge geometry
    lens = np.zeros(E, dtype=np.float32)
    cos_z = np.zeros(E, dtype=np.float32)
    for i, (a, b) in enumerate(edges):
        if a >= len(pv_arr) or b >= len(pv_arr):
            continue
        d = pv_arr[b] - pv_arr[a]
        n = float(np.linalg.norm(d))
        lens[i] = n
        if n > 1e-6:
            cos_z[i] = float(d[2] / n)
    median_len = float(np.median(lens[lens > 0])) if (lens > 0).any() else 1.0

    # Geometric core (12 dims)
    for i, (a, b) in enumerate(edges):
        if a >= len(pv_arr) or b >= len(pv_arr):
            continue
        L = lens[i]
        feats[i, 0] = float(np.log(L + 1e-3))
        feats[i, 1] = cos_z[i]
        feats[i, 2] = float(abs(cos_z[i]))
        feats[i, 3] = float(deg[a])
        feats[i, 4] = float(deg[b])
        # 5-9 filled later (image features)
        # 10-11 colmap (filled later)

    try:
        good = convert_entry_to_human_readable(sample)
        colmap_rec = good.get("colmap") or good.get("colmap_binary")
        if colmap_rec is None:
            return feats
        views = collect_views(colmap_rec, good["image_ids"])
        if not views:
            return feats
        per_view = _build_class_masks_and_corner_dt(good, views)
        if not per_view:
            return feats

        t_vec = np.linspace(0.0, 1.0, sample_steps)

        # Colmap tree (for cdist)
        pts = []
        if hasattr(colmap_rec, 'points3D'):
            for p in colmap_rec.points3D.values():
                pts.append(p.xyz)
        c_tree = cKDTree(np.asarray(pts, dtype=np.float32)) if pts else None

        for i, (u, vv) in enumerate(edges):
            u, vv = int(u), int(vv)
            if u >= len(pv_arr) or vv >= len(pv_arr) or u == vv:
                continue
            endpoints = np.stack([pv_arr[u], pv_arr[vv]])

            per_class_supp = []   # (n_views, 7)
            any_edge_supp = []    # (n_views,)
            proj_lens = []
            corner_dist_a = []
            corner_dist_b = []
            close_a = 0
            close_b = 0
            mid_edge_dist = []
            cam_z = []

            for img_id, view in views.items():
                if img_id not in per_view:
                    continue
                pv_view = per_view[img_id]
                H, W = pv_view["H"], pv_view["W"]
                uv, z = project_world_to_image(view["P"], endpoints)
                if z[0] <= 0 or z[1] <= 0:
                    continue
                if not (0 <= uv[0,0] < W and 0 <= uv[0,1] < H
                        and 0 <= uv[1,0] < W and 0 <= uv[1,1] < H):
                    continue
                cam_z.append(0.5 * (z[0] + z[1]))
                proj_lens.append(float(np.linalg.norm(uv[1] - uv[0])))

                xs = uv[0,0] + t_vec * (uv[1,0] - uv[0,0])
                ys = uv[0,1] + t_vec * (uv[1,1] - uv[0,1])
                xs_i = np.clip(xs.astype(np.int32), 0, W - 1)
                ys_i = np.clip(ys.astype(np.int32), 0, H - 1)

                # Per-class fractions
                this_view_class = np.zeros(7, dtype=np.float32)
                for ci in range(7):
                    this_view_class[ci] = float(pv_view["edge_masks"][ci][ys_i, xs_i].sum()) / sample_steps
                per_class_supp.append(this_view_class)
                any_edge_supp.append(float(this_view_class.max()))  # max-class frac as "any edge"

                # Endpoint corner distances
                cd_a = float(pv_view["corner_dt"][int(uv[0,1]), int(uv[0,0])])
                cd_b = float(pv_view["corner_dt"][int(uv[1,1]), int(uv[1,0])])
                corner_dist_a.append(cd_a)
                corner_dist_b.append(cd_b)
                if cd_a < 10.0: close_a += 1
                if cd_b < 10.0: close_b += 1

                # Midpoint edge distance
                mx, my = int(0.5 * (uv[0,0] + uv[1,0])), int(0.5 * (uv[0,1] + uv[1,1]))
                mx = max(0, min(mx, W - 1))
                my = max(0, min(my, H - 1))
                mid_edge_dist.append(float(pv_view["edge_dt"][my, mx]))

            if per_class_supp:
                arr = np.asarray(per_class_supp)  # (n_views, 7)
                feats[i, 12:19] = arr.mean(axis=0)
                feats[i, 19:26] = arr.max(axis=0)
                feats[i, 26] = float(np.mean(corner_dist_a))
                feats[i, 27] = float(np.mean(corner_dist_b))
                feats[i, 28] = float(close_a)
                feats[i, 29] = float(close_b)
                feats[i, 30] = float(np.mean(mid_edge_dist))
                any_arr = np.asarray(any_edge_supp)
                feats[i, 31] = float((any_arr > 0.3).sum())
                feats[i, 32] = float((any_arr > 0.6).sum())
                feats[i, 33] = float(np.log(max(np.mean(proj_lens), 1.0)))
                feats[i, 34] = float(any_arr.std())
                feats[i, 35] = float(np.mean(cam_z))

            # Colmap distance
            if c_tree is not None:
                a3d, b3d = pv_arr[u], pv_arr[vv]
                m3d = (a3d + b3d) * 0.5
                d_a, _ = c_tree.query(a3d)
                d_b, _ = c_tree.query(b3d)
                d_m, _ = c_tree.query(m3d)
                feats[i, 36] = float(min(d_a, d_b))
                feats[i, 37] = float(d_m)

            feats[i, 38] = float(lens[i] / max(median_len, 1e-3))
            feats[i, 39] = float(len(pv_arr) / max(median_len, 1.0))  # graph density hint

    except Exception:
        pass

    return feats


def extract_features_v2(pv, pe, sample):
    edges = [(int(a), int(b)) for a, b in pe]
    return _edge_features_v2(pv, edges, sample)


def label_edges_vs_gt(pv, pe, gt_v, gt_e, match_radius: float = 0.5):
    pv_arr = np.asarray(pv, dtype=np.float32)
    gt_v_arr = np.asarray(gt_v, dtype=np.float32)
    labels = np.zeros(len(pe), dtype=np.float32)
    if pv_arr.shape[0] < 2 or len(pe) == 0 or gt_v_arr.shape[0] < 2 or len(gt_e) == 0:
        return labels
    gt_pairs = []
    for a, b in gt_e:
        a, b = int(a), int(b)
        if 0 <= a < len(gt_v_arr) and 0 <= b < len(gt_v_arr):
            gt_pairs.append((gt_v_arr[a], gt_v_arr[b]))
    if not gt_pairs:
        return labels
    r2 = match_radius * match_radius
    for i, (u, vv) in enumerate(pe):
        u, vv = int(u), int(vv)
        if u >= len(pv_arr) or vv >= len(pv_arr):
            continue
        pa, pb = pv_arr[u], pv_arr[vv]
        for ga, gb in gt_pairs:
            d1 = ((pa - ga) ** 2).sum() + ((pb - gb) ** 2).sum()
            d2 = ((pa - gb) ** 2).sum() + ((pb - ga) ** 2).sum()
            if min(d1, d2) <= 2.0 * r2:
                labels[i] = 1.0
                break
    return labels


def classify_edges_v2(pv, pe, sample, classifier, threshold: float = 0.5,
                      feature_mean=None, feature_std=None,
                      min_keep_frac: float = 0.7):
    try:
        if len(pe) == 0:
            return pv, pe
        feats = extract_features_v2(pv, pe, sample)
        if feature_mean is not None and feature_std is not None:
            feats = (feats - feature_mean) / (feature_std + 1e-6)
        with torch.no_grad():
            x = torch.tensor(feats, dtype=torch.float32)
            scores = torch.sigmoid(classifier(x)).numpy()

        keep_mask = scores >= threshold
        min_keep = max(1, int(np.ceil(min_keep_frac * len(pe))))
        if keep_mask.sum() < min_keep:
            top_idx = np.argsort(-scores)[:min_keep]
            keep_mask = np.zeros_like(keep_mask)
            keep_mask[top_idx] = True

        keep_edges = [pe[i] for i in range(len(pe)) if keep_mask[i]]
        if len(keep_edges) < 1:
            return pv, pe

        from edge_2d_filter import drop_orphan_vertices
        return drop_orphan_vertices(np.asarray(pv), keep_edges)
    except Exception:
        return pv, pe


def load_classifier_v2(path: str, device: str = "cpu"):
    blob = torch.load(path, map_location=device, weights_only=False)
    m = EdgeClassifierV2(in_dim=blob.get("in_dim", NUM_FEATURES),
                         hidden=blob.get("hidden", 64))
    m.load_state_dict(blob["model"])
    m.to(device).eval()
    return m, blob.get("feature_mean"), blob.get("feature_std")
