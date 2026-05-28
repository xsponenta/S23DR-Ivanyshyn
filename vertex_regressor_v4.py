"""Vertex position REGRESSOR using DINOv2 features.

Predicts a 3D offset (dx, dy, dz) for each vertex to push it toward the
nearest GT vertex. This is the regression analog of the classifier — instead
of "drop bad vertices", we "move vertices to where they should be".

Loss: MSE on offset to nearest GT vertex (within match_radius), 0 if no GT
match (don't move).

Inference: clamp predicted offset magnitude to `max_move_meters` for safety.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from edge_classifier_v4 import DINO_FEAT_DIM, get_dino_model
from vertex_classifier_v4 import (
    V_GEOM_DIM, V_EDGE_FEAT_DIM,
    _vertex_geom_features, extract_vertex_features_v4,
)


class VertexRegressorV4(nn.Module):
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
            nn.Linear(hidden // 2, 3),  # (dx, dy, dz)
        )

    def forward(self, geom_feats, dino_feats):
        x = torch.cat([geom_feats, dino_feats], dim=1)
        return self.net(x)


def label_vertex_offsets(pv, gt_v, match_radius: float = 0.5):
    """Return (V, 3) target offsets: vertex moves toward nearest GT within radius.

    For vertices with no GT match within radius, target = (0, 0, 0).
    Mask (V,) is 1 where we have valid GT to learn from.
    """
    pv_arr = np.asarray(pv, dtype=np.float32)
    gt_v_arr = np.asarray(gt_v, dtype=np.float32)
    V = pv_arr.shape[0]
    offsets = np.zeros((V, 3), dtype=np.float32)
    mask = np.zeros(V, dtype=np.float32)
    if V == 0 or gt_v_arr.shape[0] == 0:
        return offsets, mask
    from scipy.spatial import cKDTree
    tree = cKDTree(gt_v_arr)
    dists, idxs = tree.query(pv_arr)
    keep = dists <= match_radius
    offsets[keep] = gt_v_arr[idxs[keep]] - pv_arr[keep]
    mask[keep] = 1.0
    return offsets, mask


def refine_vertices_with_regressor(pv, pe, sample, regressor, dino, device="cpu",
                                   feature_mean=None, feature_std=None,
                                   edge_feat_mean=None, edge_feat_std=None,
                                   max_move_meters: float = 0.4,
                                   min_views_for_move: int = 2):
    """Apply regressor to predict per-vertex offsets, clamp by max_move."""
    try:
        pv_arr = np.asarray(pv, dtype=np.float32)
        if pv_arr.shape[0] == 0:
            return pv, pe
        geom, dino_feats = extract_vertex_features_v4(pv, pe, sample, dino, device=device)
        # Only move vertices visible in enough views (sanity)
        in_view_count = geom[:, 4].astype(np.int32)  # feature[4] is num views vertex is in-frame

        if feature_mean is not None and feature_std is not None:
            geom_n = (geom - feature_mean) / (feature_std + 1e-6)
        else:
            geom_n = geom
        if edge_feat_mean is not None and edge_feat_std is not None:
            dino_n = (dino_feats - edge_feat_mean) / (edge_feat_std + 1e-6)
        else:
            dino_n = dino_feats

        with torch.no_grad():
            g = torch.tensor(geom_n, dtype=torch.float32)
            d = torch.tensor(dino_n, dtype=torch.float32)
            offsets = regressor(g, d).numpy()  # (V, 3)

        # Clamp magnitude
        norms = np.linalg.norm(offsets, axis=1, keepdims=True)
        scale = np.clip(max_move_meters / np.maximum(norms, 1e-9), 0, 1)
        clamped = offsets * scale

        # Don't move vertices with too few view supports
        no_move = in_view_count < min_views_for_move
        clamped[no_move] = 0

        new_pv = pv_arr + clamped
        return new_pv.astype(np.float64), pe
    except Exception:
        return pv, pe


def load_regressor_v4(path: str, device: str = "cpu"):
    blob = torch.load(path, map_location=device, weights_only=False)
    m = VertexRegressorV4(hidden=blob.get("hidden", 128))
    m.load_state_dict(blob["model"])
    m.to(device).eval()
    return (m, blob.get("feature_mean"), blob.get("feature_std"),
            blob.get("edge_feat_mean"), blob.get("edge_feat_std"))
