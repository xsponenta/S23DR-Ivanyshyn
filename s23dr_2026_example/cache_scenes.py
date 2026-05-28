#!/usr/bin/env python3
"""Cache compact scenes from HoHo22k shards to training-ready .pt files.

Streams samples from the public `usm3d/hoho22k_2026_trainval` dataset, runs
`build_compact_scene` (see point_fusion.py), precomputes priority group_id
and semantic class_id, and saves one .pt per scene.

Stage 1 of the dataset pipeline. See make_sampled_cache.py for stage 2.

Usage:
  python -m s23dr_2026_example.cache_scenes --out-dir cache/full --split train
  python -m s23dr_2026_example.cache_scenes --out-dir cache/full_val --split validation

Cache format per .pt file:
  xyz:             float32 [P, 3]   all points in world space
  source:          uint8   [P]      0=colmap, 1=depth
  group_id:        int8    [P]      priority tier 0-4, -1=excluded
  class_id:        uint8   [P]      one-hot class index (0-12)
  behind_gest_id:  int16   [P]      behind-gestalt id (-1 if none)
  visible_src:     uint8   [P]      1=gestalt, 2=ade
  visible_id:      int16   [P]      class id within space
  n_views_voted:   uint8   [P]      number of views that voted
  vote_frac:       float32 [P]      fraction of votes
  center:          float32 [3]      smart normalization center
  scale:           float32 scalar   smart normalization scale
  gt_vertices:     float32 [V, 3]   ground truth wireframe vertices
  gt_edges:        int32   [E, 2]   ground truth wireframe edge indices
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .point_fusion import (
    FuserConfig, build_compact_scene,
    GEST_ID_TO_NAME, ADE_ID_TO_NAME, NUM_GEST,
)

# ---------------------------------------------------------------------------
# Semantic class encoding: 11 structural + 1 other_house + 1 non_house = 13
# ---------------------------------------------------------------------------

# Each structural gestalt class gets its own one-hot bit.
STRUCTURAL_CLASSES = (
    "apex", "eave_end_point", "flashing_end_point",     # point classes (tier 0)
    "rake", "ridge", "eave", "hip", "valley",           # roof edges (tier 1)
    "flashing", "step_flashing",
    "roof",                                              # roof face (tier 2)
)
# Index 11 = other house part (door, window, siding, etc.)
# Index 12 = non-house / ADE / unlabeled
NUM_SEMANTIC_CLASSES = len(STRUCTURAL_CLASSES) + 2  # 13

# Priority tiers (same as tokenizer.py)
_GEST_NAME_TO_ID = {n: i for i, n in enumerate(GEST_ID_TO_NAME)}
_POINT_IDS = {_GEST_NAME_TO_ID[n] for n in ("apex", "eave_end_point", "flashing_end_point") if n in _GEST_NAME_TO_ID}
_EDGE_IDS = {_GEST_NAME_TO_ID[n] for n in ("rake", "ridge", "eave", "hip", "valley", "flashing", "step_flashing") if n in _GEST_NAME_TO_ID}
_FACE_IDS = {_GEST_NAME_TO_ID[n] for n in ("roof",) if n in _GEST_NAME_TO_ID}
_HOUSE_IDS = {_GEST_NAME_TO_ID[n] for n in (
    "apex", "eave_end_point", "flashing_end_point",
    "rake", "ridge", "eave", "hip", "valley", "flashing", "step_flashing",
    "roof", "door", "garage", "window", "shutter", "fascia", "soffit",
    "horizontal_siding", "vertical_siding", "brick", "concrete",
    "other_wall", "trim", "post", "ground_line",
) if n in _GEST_NAME_TO_ID}

_ADE_NAME_TO_ID = {n.lower(): i for i, n in enumerate(ADE_ID_TO_NAME)}
_ADE_HOUSE_IDS = {_ADE_NAME_TO_ID[n] for n in ("building;edifice", "house", "wall", "windowpane;window", "door;double;door") if n in _ADE_NAME_TO_ID}

_UNCLS_ID = _GEST_NAME_TO_ID.get("unclassified", -1)

# Map structural gestalt names to one-hot index
_STRUCTURAL_ONEHOT = {}
for idx, name in enumerate(STRUCTURAL_CLASSES):
    gid = _GEST_NAME_TO_ID.get(name)
    if gid is not None:
        _STRUCTURAL_ONEHOT[gid] = idx


def _compute_group_and_class(visible_src, visible_id, behind_id, source):
    """Compute priority group_id and semantic class_id per point (vectorized).

    Args:
        visible_src: uint8 [P] -- 0=unlabeled, 1=gestalt, 2=ade
        visible_id:  int16 [P] -- class id within gestalt or ade space
        behind_id:   int16 [P] -- behind-gestalt id (-1 if none)
        source:      uint8 [P] -- 0=colmap, 1=depth

    Returns:
        group_id:  int8  [P] -- priority tier 0-4, -1 for excluded (unclassified)
        class_id:  uint8 [P] -- one-hot class index 0-12
    """
    P = len(visible_src)
    vsrc = visible_src.astype(np.int32)
    vid = visible_id.astype(np.int32)
    bid = behind_id.astype(np.int32)

    # Effective gestalt id: prefer visible gestalt, fall back to behind
    gest_id = np.full(P, -1, dtype=np.int32)
    has_vis_gest = (vsrc == 1) & (vid >= 0)
    has_behind = (bid >= 0) & ~has_vis_gest
    gest_id[has_vis_gest] = vid[has_vis_gest]
    gest_id[has_behind] = bid[has_behind]

    # Exclude unclassified points
    if _UNCLS_ID >= 0:
        is_uncls = ((vsrc == 1) & (vid == _UNCLS_ID)) | (bid == _UNCLS_ID)
        gest_id[is_uncls] = -1  # force excluded

    # Build lookup arrays for gestalt id -> group and gestalt id -> class
    max_gid = NUM_GEST
    gid_to_group = np.full(max_gid, 4, dtype=np.int8)  # default: tier 4
    gid_to_class = np.full(max_gid, NUM_SEMANTIC_CLASSES - 1, dtype=np.uint8)  # default: non-house

    for gid in _POINT_IDS:
        gid_to_group[gid] = 0
    for gid in _EDGE_IDS:
        gid_to_group[gid] = 1
    for gid in _FACE_IDS:
        gid_to_group[gid] = 2
    for gid in _HOUSE_IDS - _POINT_IDS - _EDGE_IDS - _FACE_IDS:
        gid_to_group[gid] = 3
    for gid, onehot_idx in _STRUCTURAL_ONEHOT.items():
        gid_to_class[gid] = onehot_idx
    for gid in _HOUSE_IDS - set(_STRUCTURAL_ONEHOT.keys()):
        gid_to_class[gid] = len(STRUCTURAL_CLASSES)  # other_house

    # Apply lookup for points with valid gestalt ids
    has_gest = gest_id >= 0
    group_id = np.full(P, 4, dtype=np.int8)  # default: tier 4
    class_id = np.full(P, NUM_SEMANTIC_CLASSES - 1, dtype=np.uint8)  # default: non-house

    group_id[has_gest] = gid_to_group[gest_id[has_gest]]
    class_id[has_gest] = gid_to_class[gest_id[has_gest]]

    # ADE house points (no gestalt) get tier 3 + class_id = other_house
    ade_house_arr = np.array(sorted(_ADE_HOUSE_IDS), dtype=np.int32)
    is_ade_house = ~has_gest & (vsrc == 2) & (vid >= 0) & np.isin(vid, ade_house_arr)
    group_id[is_ade_house] = 3
    class_id[is_ade_house] = len(STRUCTURAL_CLASSES)  # other_house (index 11)

    # Mark excluded points (unclassified) as -1
    if _UNCLS_ID >= 0:
        group_id[is_uncls] = -1
        class_id[is_uncls] = NUM_SEMANTIC_CLASSES - 1

    return group_id, class_id


def _compute_smart_center_scale(xyz, source, mad_k=2.5, percentile=95.0,
                                 max_points=8000):
    """Compute normalization center and scale from depth points with MAD filter."""
    depth_mask = source == 1
    ref = xyz[depth_mask] if depth_mask.any() else xyz
    if ref.shape[0] == 0:
        center = xyz.mean(axis=0)
        scale = max(np.linalg.norm(xyz - center, axis=1).max(), 1e-6)
        return center.astype(np.float32), np.float32(scale)

    if ref.shape[0] > max_points:
        idx = np.random.choice(ref.shape[0], max_points, replace=False)
        ref = ref[idx]

    center0 = np.median(ref, axis=0)
    dist = np.linalg.norm(ref - center0, axis=1)
    med = np.median(dist)
    mad = max(np.median(np.abs(dist - med)), 1e-6)
    inliers = dist <= (med + mad_k * mad)
    if inliers.any():
        ref = ref[inliers]

    # Percentile bounding box
    lo_f = (100.0 - percentile) * 0.5 / 100.0
    sorted_v = np.sort(ref, axis=0)
    n = sorted_v.shape[0]
    lo_idx = max(0, min(n - 1, int(lo_f * (n - 1))))
    hi_idx = max(0, min(n - 1, int((1.0 - lo_f) * (n - 1))))
    low = sorted_v[lo_idx]
    high = sorted_v[hi_idx]

    center = 0.5 * (low + high)
    scale = max(np.sqrt(((high - low) ** 2).sum()), 1e-6)
    return center.astype(np.float32), np.float32(scale)


# ---------------------------------------------------------------------------
# Dataset pipeline stage 1: raw HF sample -> cached .pt
# ---------------------------------------------------------------------------

def _process_one(sample, cfg):
    """Fuse a single HF sample into a cache dict. Returns (order_id, dict) or None."""
    rng = np.random.RandomState()

    n_edges = len(sample.get("wf_edges", []))
    if n_edges == 0 or n_edges > 64:
        return None

    scene = build_compact_scene(sample, cfg, rng=rng)
    if scene is None:
        return None

    gt_v = scene.get("gt_vertices")
    gt_e = scene.get("gt_edges")
    if gt_v is None or gt_e is None or len(gt_e) == 0:
        return None

    xyz = scene["xyz"]
    source = scene["source"]
    group_id, class_id = _compute_group_and_class(
        scene["visible_src"], scene["visible_id"], scene["behind_gest_id"], source)
    center, scale = _compute_smart_center_scale(xyz, source)

    gt_edge_classes = np.asarray(sample["wf_classifications"], dtype=np.int64)
    return sample["order_id"], {
        "xyz": xyz.astype(np.float32),
        "source": source.astype(np.uint8),
        "group_id": group_id,
        "class_id": class_id,
        "behind_gest_id": scene["behind_gest_id"].astype(np.int16),
        "visible_src": scene["visible_src"].astype(np.uint8),
        "visible_id": scene["visible_id"].astype(np.int16),
        "n_views_voted": scene["n_views_voted"],
        "vote_frac": scene["vote_frac"],
        "center": center,
        "scale": scale,
        "gt_vertices": gt_v.astype(np.float32),
        "gt_edges": gt_e.astype(np.int32),
        "gt_edge_classes": gt_edge_classes,
    }


def main():
    p = argparse.ArgumentParser(description="Stage 1: HoHo22k -> cached .pt files")
    p.add_argument("--out-dir", required=True, help="Output directory for .pt files")
    p.add_argument("--split", default="train", choices=["train", "validation"])
    p.add_argument("--limit", type=int, default=0, help="Stop after N samples (0 = all)")
    p.add_argument("--depth-per-view", type=int, default=8000)
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = {p.stem for p in out_dir.glob("*.pt")} if args.skip_existing else set()

    from datasets import load_dataset
    print(f"Streaming usm3d/hoho22k_2026_trainval split={args.split}...")
    ds = load_dataset("usm3d/hoho22k_2026_trainval",
                      streaming=True, trust_remote_code=True, split=args.split)

    cfg = FuserConfig(depth_points_per_view=args.depth_per_view)
    saved, skipped = 0, 0
    t0 = time.perf_counter()
    for i, sample in enumerate(ds):
        if args.limit > 0 and i >= args.limit:
            break
        oid = sample["order_id"]
        if oid in existing:
            skipped += 1
            continue
        result = _process_one(sample, cfg)
        if result is None:
            skipped += 1
            continue
        order_id, data = result
        torch.save(data, out_dir / f"{order_id}.pt")
        saved += 1
        if saved % 100 == 0:
            rate = saved / (time.perf_counter() - t0)
            print(f"  saved {saved} (skipped {skipped}) [{rate:.1f}/s]")

    elapsed = time.perf_counter() - t0
    print(f"Done. Saved {saved}, skipped {skipped} in {elapsed:.0f}s.")


if __name__ == "__main__":
    main()


