"""V4 vertex classifier with DINOv2 patch features.

Per-vertex analog of edge_classifier_v4. For each predicted 3D vertex,
project to each view, bilinearly sample DINOv2 patch features at the
projected pixel location. Mean+max pool across views -> 768-dim. Concat
with simple geometric features (10-dim) -> 778-dim. MLP head -> P(keep).

Decisions:
  - drop a vertex if classifier P(keep) < threshold
  - re-run drop_orphan after to clean dangling edges

Label for training: vertex is "true" if there's a GT vertex within
`match_radius` meters.

Vertex geometric features (10):
   0  z (height)
   1  degree (number of incident edges)
   2  dist to nearest other vertex
   3  dist to nearest colmap point
   4  num views vertex is in-frame
   5  min projected dist to nearest gestalt corner pixel (clipped 30)
   6  mean projected dist to nearest gestalt corner pixel
   7  num views with corner pixel within 5px
   8  num views with corner pixel within 15px
   9  vertex_count / median scene vertex count (graph density hint)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from edge_classifier_v4 import (
    DINO_FEAT_DIM, get_dino_model, _encode_views_with_dino, _bilinear_sample_grid,
)

POINT_CLASSES = ("apex", "eave_end_point", "flashing_end_point")
V_GEOM_DIM = 10
V_EDGE_FEAT_DIM = DINO_FEAT_DIM * 2  # mean + max pool


class VertexClassifierV4(nn.Module):
    def __init__(self, geom_dim: int = V_GEOM_DIM, edge_feat_dim: int = V_EDGE_FEAT_DIM,
                 hidden: int = 128):
        super().__init__()
        in_dim = geom_dim + edge_feat_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, geom_feats, dino_feats):
        x = torch.cat([geom_feats, dino_feats], dim=1)
        return self.net(x).squeeze(-1)


def _build_corner_dt(good, views, dilate_px=0):
    """Per-view corner DT (distance to nearest gestalt point-class pixel)."""
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
        mask = np.zeros((H, W), dtype=np.uint8)
        for cls in POINT_CLASSES:
            color = np.array(gestalt_color_mapping[cls])
            mask |= cv2.inRange(gest_np, color - 0.5, color + 0.5)
        if mask.sum() > 0:
            dt = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5)
            dt = np.minimum(dt, 30.0).astype(np.float32)
        else:
            dt = np.full((H, W), 30.0, dtype=np.float32)
        out[img_id] = (dt, H, W)
    return out


def _vertex_geom_features(pv, pe, sample):
    """Per-vertex geometric features (V, 10)."""
    pv_arr = np.asarray(pv, dtype=np.float32)
    V = len(pv_arr)
    feats = np.zeros((V, V_GEOM_DIM), dtype=np.float32)
    if V == 0:
        return feats

    deg = np.zeros(V, dtype=np.int32)
    for a, b in pe:
        if 0 <= int(a) < V: deg[int(a)] += 1
        if 0 <= int(b) < V: deg[int(b)] += 1

    try:
        from hoho2025.example_solutions import convert_entry_to_human_readable
        from mvs_utils import collect_views, project_world_to_image
        from scipy.spatial import cKDTree

        good = convert_entry_to_human_readable(sample)
        colmap_rec = good.get("colmap") or good.get("colmap_binary")
        if colmap_rec is None:
            return feats
        views = collect_views(colmap_rec, good["image_ids"])
        if not views:
            return feats
        corner_dt = _build_corner_dt(good, views)

        # KDTree over colmap points
        c_pts = []
        if hasattr(colmap_rec, "points3D"):
            for p in colmap_rec.points3D.values():
                c_pts.append(p.xyz)
        c_tree = cKDTree(np.asarray(c_pts, dtype=np.float32)) if c_pts else None

        # KDTree over pred vertices (for nearest-neighbor)
        pv_tree = cKDTree(pv_arr) if V >= 2 else None

        for i in range(V):
            v = pv_arr[i]
            feats[i, 0] = float(v[2])
            feats[i, 1] = float(deg[i])
            if pv_tree is not None:
                dists, _ = pv_tree.query(v, k=min(2, V))
                feats[i, 2] = float(dists[1] if len(dists) > 1 else 0.0)
            if c_tree is not None:
                d, _ = c_tree.query(v)
                feats[i, 3] = float(d)

            # Per-view corner distances
            view_dists = []
            in_frame = 0
            close_5 = 0
            close_15 = 0
            for img_id, view in views.items():
                if img_id not in corner_dt:
                    continue
                dt, H, W = corner_dt[img_id]
                uv, z = project_world_to_image(view["P"], v.reshape(1, 3))
                if z[0] <= 0:
                    continue
                if not (0 <= uv[0, 0] < W and 0 <= uv[0, 1] < H):
                    continue
                in_frame += 1
                cd = float(dt[int(uv[0, 1]), int(uv[0, 0])])
                view_dists.append(cd)
                if cd < 5: close_5 += 1
                if cd < 15: close_15 += 1
            feats[i, 4] = float(in_frame)
            if view_dists:
                arr = np.asarray(view_dists)
                feats[i, 5] = float(arr.min())
                feats[i, 6] = float(arr.mean())
            else:
                feats[i, 5] = 30.0
                feats[i, 6] = 30.0
            feats[i, 7] = float(close_5)
            feats[i, 8] = float(close_15)
            feats[i, 9] = float(V)
    except Exception:
        pass
    return feats


def extract_vertex_features_v4(pv, pe, sample, dino, device="cpu"):
    """Return (V, 10) geom + (V, 768) dino features per vertex."""
    pv_arr = np.asarray(pv)
    V = len(pv_arr)
    geom = _vertex_geom_features(pv, pe, sample)
    dino_feats = np.zeros((V, V_EDGE_FEAT_DIM), dtype=np.float32)
    if V == 0:
        return geom, dino_feats

    try:
        from hoho2025.example_solutions import convert_entry_to_human_readable
        from mvs_utils import collect_views, project_world_to_image
        good = convert_entry_to_human_readable(sample)
        colmap_rec = good.get("colmap") or good.get("colmap_binary")
        if colmap_rec is None:
            return geom, dino_feats
        views = collect_views(colmap_rec, good["image_ids"])
        if not views:
            return geom, dino_feats
        per_view, Hs, Ws = _encode_views_with_dino(good, views, dino, device)
        if not per_view:
            return geom, dino_feats

        for i in range(V):
            v = pv_arr[i].reshape(1, 3)
            per_view_feats = []
            for img_id, view in views.items():
                if img_id not in per_view:
                    continue
                H, W = Hs[img_id], Ws[img_id]
                uv, z = project_world_to_image(view["P"], v)
                if z[0] <= 0:
                    continue
                u, vu = uv[0, 0], uv[0, 1]
                if not (0 <= u < W and 0 <= vu < H):
                    continue
                feat = _bilinear_sample_grid(per_view[img_id], u / max(W - 1, 1), vu / max(H - 1, 1))
                per_view_feats.append(feat)
            if per_view_feats:
                arr = np.asarray(per_view_feats)
                dino_feats[i] = np.concatenate([arr.mean(axis=0), arr.max(axis=0)])
    except Exception:
        pass
    return geom, dino_feats


def label_vertices_vs_gt(pv, gt_v, match_radius: float = 0.5):
    """Vertex is positive if a GT vertex is within match_radius meters."""
    pv_arr = np.asarray(pv, dtype=np.float32)
    gt_v_arr = np.asarray(gt_v, dtype=np.float32)
    if pv_arr.shape[0] == 0 or gt_v_arr.shape[0] == 0:
        return np.zeros(pv_arr.shape[0], dtype=np.float32)
    from scipy.spatial import cKDTree
    tree = cKDTree(gt_v_arr)
    dists, _ = tree.query(pv_arr)
    return (dists <= match_radius).astype(np.float32)


def classify_vertices_v4(pv, pe, sample, classifier, dino, device="cpu",
                         threshold: float = 0.5,
                         feature_mean=None, feature_std=None,
                         edge_feat_mean=None, edge_feat_std=None,
                         min_keep_frac: float = 0.85):
    """Drop low-confidence vertices, rebuild edge indexing."""
    try:
        pv_arr = np.asarray(pv)
        if len(pv_arr) == 0:
            return pv, pe
        geom, dino_feats = extract_vertex_features_v4(pv, pe, sample, dino, device=device)
        if feature_mean is not None and feature_std is not None:
            geom = (geom - feature_mean) / (feature_std + 1e-6)
        if edge_feat_mean is not None and edge_feat_std is not None:
            dino_feats = (dino_feats - edge_feat_mean) / (edge_feat_std + 1e-6)
        with torch.no_grad():
            g = torch.tensor(geom, dtype=torch.float32)
            d = torch.tensor(dino_feats, dtype=torch.float32)
            scores = torch.sigmoid(classifier(g, d)).numpy()

        keep_mask = scores >= threshold
        min_keep = max(2, int(np.ceil(min_keep_frac * len(pv_arr))))
        if keep_mask.sum() < min_keep:
            top_idx = np.argsort(-scores)[:min_keep]
            keep_mask = np.zeros_like(keep_mask)
            keep_mask[top_idx] = True

        if keep_mask.all():
            return pv, pe

        keep_idx = np.where(keep_mask)[0]
        old_to_new = {int(o): int(n) for n, o in enumerate(keep_idx)}
        new_pv = pv_arr[keep_idx]
        new_pe = [(old_to_new[int(a)], old_to_new[int(b)])
                  for a, b in pe
                  if int(a) in old_to_new and int(b) in old_to_new]
        if len(new_pv) < 2 or len(new_pe) < 1:
            return pv, pe
        return new_pv, new_pe
    except Exception:
        return pv, pe


def load_classifier_v4(path: str, device: str = "cpu"):
    blob = torch.load(path, map_location=device, weights_only=False)
    m = VertexClassifierV4(hidden=blob.get("hidden", 128))
    m.load_state_dict(blob["model"])
    m.to(device).eval()
    return (m, blob.get("feature_mean"), blob.get("feature_std"),
            blob.get("edge_feat_mean"), blob.get("edge_feat_std"))
