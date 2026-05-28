"""V4 edge classifier with DINOv2-pretrained patch features.

DINOv2-small (22M params, frozen) encodes each gestalt view to a 16x16 grid
of 384-dim semantic features. For each predicted edge, we bilinearly sample
features at the projected edge midpoint in each view, then mean+max pool
across views to get a 768-dim feature per edge. Concatenated with v2's
40-dim geometric+mask features = 808-dim input to MLP.

DINOv2 patch size is 14, input 224x224 → 16x16 patches. We feed gestalt RGB
resized to 224x224.

Falls back to no-op on any error.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from edge_classifier_v2 import (
    NUM_FEATURES as V2_NUM, extract_features_v2,
)

DINO_FEAT_DIM = 384
DINO_PATCH_SIZE = 14
DINO_INPUT = 224
DINO_GRID = DINO_INPUT // DINO_PATCH_SIZE  # 16
EDGE_FEAT_DIM = DINO_FEAT_DIM * 2  # mean + max pool across views


def get_dino_model(device="cpu"):
    """Load DINOv2-small (frozen). Cached after first call."""
    if not hasattr(get_dino_model, "_cache"):
        get_dino_model._cache = {}
    cache_key = str(device)
    if cache_key not in get_dino_model._cache:
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', verbose=False)
        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad = False
        get_dino_model._cache[cache_key] = model
    return get_dino_model._cache[cache_key]


class EdgeClassifierV4(nn.Module):
    """Head: takes pre-pooled DINOv2 edge features + v2 geom features."""
    def __init__(self, geom_dim: int = V2_NUM, edge_feat_dim: int = EDGE_FEAT_DIM,
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

    def forward(self, geom_feats, edge_feats):
        x = torch.cat([geom_feats, edge_feats], dim=1)
        return self.net(x).squeeze(-1)


@torch.no_grad()
def _encode_views_with_dino(good, views, dino, device):
    """Encode each view's gestalt image with DINOv2. Returns dict img_id -> (16,16,384)."""
    out = {}
    imgs = []
    img_ids = []
    Hs, Ws = {}, {}
    for gest_pil, depth_pil, img_id in zip(
        good["gestalt"], good["depth"], good["image_ids"]
    ):
        if img_id not in views:
            continue
        depth_np = np.array(depth_pil)
        H, W = depth_np.shape[:2]
        # Resize gestalt to 224x224 (DINOv2 input)
        gest_resized = np.array(gest_pil.resize((DINO_INPUT, DINO_INPUT))).astype(np.float32) / 255.0
        if gest_resized.ndim == 2:
            gest_resized = np.stack([gest_resized]*3, axis=-1)
        else:
            gest_resized = gest_resized[..., :3]
        # ImageNet normalization
        gest_resized = (gest_resized - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        imgs.append(gest_resized.transpose(2, 0, 1))  # (3, 224, 224)
        img_ids.append(img_id)
        Hs[img_id] = H
        Ws[img_id] = W
        # Inject H,W for downstream code
        views[img_id]["H"] = H
        views[img_id]["W"] = W

    if not imgs:
        return out, Hs, Ws

    batch = torch.tensor(np.stack(imgs), dtype=torch.float32, device=device)
    # Forward (batch)
    feats = dino.forward_features(batch)
    patches = feats["x_norm_patchtokens"]  # (B, 256, 384)
    patches = patches.reshape(len(imgs), DINO_GRID, DINO_GRID, DINO_FEAT_DIM).cpu().numpy()
    for i, img_id in enumerate(img_ids):
        out[img_id] = patches[i]  # (16, 16, 384)
    return out, Hs, Ws


def _bilinear_sample_grid(grid, u_norm, v_norm):
    """Bilinearly sample (G, G, D) grid at (u, v) in [0, 1] coords. Returns (D,)."""
    G = grid.shape[0]
    u = u_norm * (G - 1)
    v = v_norm * (G - 1)
    x0 = int(np.floor(u)); x1 = min(x0 + 1, G - 1)
    y0 = int(np.floor(v)); y1 = min(y0 + 1, G - 1)
    x0 = max(0, x0); y0 = max(0, y0)
    dx = u - x0; dy = v - y0
    f00 = grid[y0, x0]
    f01 = grid[y0, x1]
    f10 = grid[y1, x0]
    f11 = grid[y1, x1]
    f0 = f00 * (1 - dx) + f01 * dx
    f1 = f10 * (1 - dx) + f11 * dx
    return f0 * (1 - dy) + f1 * dy


def extract_features_v4(pv, pe, sample, dino, device="cpu"):
    """Return (E, geom_dim), (E, edge_feat_dim) tensors."""
    pv_arr = np.asarray(pv)
    E = len(pe)
    geom = extract_features_v2(pv, pe, sample)
    edge_feats = np.zeros((E, EDGE_FEAT_DIM), dtype=np.float32)
    if E == 0:
        return geom, edge_feats

    try:
        from hoho2025.example_solutions import convert_entry_to_human_readable
        from mvs_utils import collect_views, project_world_to_image
        good = convert_entry_to_human_readable(sample)
        colmap_rec = good.get("colmap") or good.get("colmap_binary")
        if colmap_rec is None:
            return geom, edge_feats
        views = collect_views(colmap_rec, good["image_ids"])
        if not views:
            return geom, edge_feats
        per_view, Hs, Ws = _encode_views_with_dino(good, views, dino, device)
        if not per_view:
            return geom, edge_feats

        for i, (u, vv) in enumerate(pe):
            u, vv = int(u), int(vv)
            if u >= len(pv_arr) or vv >= len(pv_arr) or u == vv:
                continue
            endpoints = np.stack([pv_arr[u], pv_arr[vv]])
            per_view_feats = []
            for img_id, view in views.items():
                if img_id not in per_view:
                    continue
                H, W = Hs[img_id], Ws[img_id]
                uv, z = project_world_to_image(view["P"], endpoints)
                if z[0] <= 0 or z[1] <= 0:
                    continue
                # Midpoint
                mx, my = 0.5 * (uv[0, 0] + uv[1, 0]), 0.5 * (uv[0, 1] + uv[1, 1])
                if not (0 <= mx < W and 0 <= my < H):
                    continue
                # Sample DINOv2 grid at (mx/W, my/H)
                feat = _bilinear_sample_grid(per_view[img_id], mx / max(W - 1, 1), my / max(H - 1, 1))
                per_view_feats.append(feat)
            if per_view_feats:
                arr = np.asarray(per_view_feats)  # (V, 384)
                mean_f = arr.mean(axis=0)
                max_f = arr.max(axis=0)
                edge_feats[i] = np.concatenate([mean_f, max_f])
    except Exception:
        pass

    return geom, edge_feats


def label_edges_vs_gt(pv, pe, gt_v, gt_e, match_radius: float = 0.4):
    from edge_classifier_v2 import label_edges_vs_gt as _f
    return _f(pv, pe, gt_v, gt_e, match_radius)


def classify_edges_v4(pv, pe, sample, classifier, dino, device="cpu",
                     threshold: float = 0.5,
                     feature_mean=None, feature_std=None,
                     edge_feat_mean=None, edge_feat_std=None,
                     min_keep_frac: float = 0.7):
    try:
        if len(pe) == 0:
            return pv, pe
        geom, edge_feats = extract_features_v4(pv, pe, sample, dino, device=device)
        if feature_mean is not None and feature_std is not None:
            geom = (geom - feature_mean) / (feature_std + 1e-6)
        if edge_feat_mean is not None and edge_feat_std is not None:
            edge_feats = (edge_feats - edge_feat_mean) / (edge_feat_std + 1e-6)
        with torch.no_grad():
            g = torch.tensor(geom, dtype=torch.float32, device=device)
            e = torch.tensor(edge_feats, dtype=torch.float32, device=device)
            scores = torch.sigmoid(classifier(g, e)).cpu().numpy()

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


def load_classifier_v4(path: str, device: str = "cpu"):
    blob = torch.load(path, map_location=device, weights_only=False)
    m = EdgeClassifierV4(geom_dim=V2_NUM, edge_feat_dim=EDGE_FEAT_DIM,
                         hidden=blob.get("hidden", 128))
    m.load_state_dict(blob["model"])
    m.to(device).eval()
    return (m, blob.get("feature_mean"), blob.get("feature_std"),
            blob.get("edge_feat_mean"), blob.get("edge_feat_std"))
