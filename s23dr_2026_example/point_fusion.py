"""
point_fusion.py

Simplified semantic point fusion for the 2026 dataset format.

Takes per-view (ADE segmap, Gestalt segmap, depth) + sparse COLMAP point cloud
from the usm3d/hoho22k_2026_trainval dataset and builds a compact, house-centric
semantic point representation suitable for downstream wireframe prediction.

Key differences from the 2025 pipeline:
  - COLMAP is a ZIP of text files (cameras.txt, images.txt, points3D.txt)
  - Depth is millimeter I;16 PNG (depth_scale=0.001 converts to meters)
  - Views flagged with pose_only_in_colmap=True have zeroed K/R/t and must be
    skipped for depth unprojection and projection
  - Images arrive as PIL Images, not byte arrays
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.stats import mode as scipy_mode

from .color_mappings import ade20k_color_mapping, gestalt_color_mapping

# ---------------------------------------------------------------------------
# Color packing helpers
# ---------------------------------------------------------------------------

def _pack_rgb_u32(rgb: np.ndarray) -> np.ndarray:
    """Pack uint8 RGB (..., 3) into uint32 codes."""
    rgb = rgb.astype(np.uint32, copy=False)
    return (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]


def _build_rgbcode_maps(color_mapping):
    """Return (rgbcode_to_id, id_to_name) for a color mapping dict."""
    names = list(color_mapping.keys())
    rgbs = np.array([color_mapping[n] for n in names], dtype=np.uint8)
    codes = _pack_rgb_u32(rgbs.reshape(-1, 1, 3)).reshape(-1)
    rgbcode_to_id = {int(c): i for i, c in enumerate(codes)}
    return rgbcode_to_id, names


def _name_to_packed_rgb(name, mapping):
    """Case-insensitive lookup returning a packed RGB code, or None."""
    for key in mapping:
        if key.lower() == name.lower():
            rgb = np.array(mapping[key], np.uint8).reshape(1, 1, 3)
            return int(_pack_rgb_u32(rgb).reshape(()))
    return None

# ---------------------------------------------------------------------------
# Label mapping constants
# ---------------------------------------------------------------------------

ADE_RGBCODE_TO_ID, ADE_ID_TO_NAME = _build_rgbcode_maps(ade20k_color_mapping)
GEST_RGBCODE_TO_ID, GEST_ID_TO_NAME = _build_rgbcode_maps(gestalt_color_mapping)
NUM_ADE = len(ADE_ID_TO_NAME)
NUM_GEST = len(GEST_ID_TO_NAME)

GEST_INVALID_NAMES = ("unclassified", "unknown", "transition_line")
GEST_INVALID_CODES = set(
    int(_pack_rgb_u32(np.array(gestalt_color_mapping[n], np.uint8).reshape(1, 1, 3)).reshape(()))
    for n in GEST_INVALID_NAMES if n in gestalt_color_mapping
)

# ADE classes whose surfaces are "see-through" for label fusion: when a point
# projects onto one of these, we use the Gestalt label behind it instead.
ADE_TRANSPARENT_NAMES = (
    "wall", "building;edifice", "floor;flooring", "ceiling",
    "windowpane;window", "door;double;door", "house", "skyscraper",
    "screen;door;screen", "blind;screen", "hovel;hut;hutch;shack;shanty",
    "tower", "booth;cubicle;stall;kiosk",
)

# ADE classes kept as "occluders/add-ons" when overlapping the house silhouette.
ADE_OCCLUDER_ALLOWLIST_NAMES = (
    "tree", "person;individual;someone;somebody;mortal;soul",
    "car;auto;automobile;machine;motorcar", "truck;motortruck", "van",
    "fence;fencing", "railing;rail",
    "bannister;banister;balustrade;balusters;handrail",
    "stairs;steps", "stairway;staircase", "step;stair", "pole",
    "streetlight;street;lamp", "signboard;sign", "awning;sunshade;sunblind",
    "plant;flora;plant;life", "pot;flowerpot",
)

# Precomputed arrays for the default name lists (avoids re-lookup every call).
_DEFAULT_ADE_TRANSPARENT_CODES = np.array(
    [c for n in ADE_TRANSPARENT_NAMES
     if (c := _name_to_packed_rgb(n, ade20k_color_mapping)) is not None],
    dtype=np.uint32,
)
_DEFAULT_ADE_OCCLUDER_IDS = np.array(
    sorted({ADE_RGBCODE_TO_ID[c]
            for n in ADE_OCCLUDER_ALLOWLIST_NAMES
            if (c := _name_to_packed_rgb(n, ade20k_color_mapping)) is not None
            and c in ADE_RGBCODE_TO_ID}),
    dtype=np.int32,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FuserConfig:
    """Simplified fusion configuration (no depth calibration fields)."""
    depth_points_per_view: int = 20_000          # depth samples per view
    depth_scale: float = 0.001                   # mm -> meters
    depth_clip_percentile: float = 99.5          # drop extreme outliers
    house_mask_dilate_px: int = 5                # dilate gestalt mask
    min_support_views: int = 1                   # min views for a kept point
    ade_transparent_classes: Tuple[str, ...] = ADE_TRANSPARENT_NAMES
    ade_occluder_allowlist: Tuple[str, ...] = ADE_OCCLUDER_ALLOWLIST_NAMES

# ---------------------------------------------------------------------------
# Geometry: projection + depth unprojection
# ---------------------------------------------------------------------------

def project_world_points(points_world, K, R, t):
    """Project (N,3) world points to pixel (u,v) with validity mask."""
    pts = points_world.astype(np.float32, copy=False)
    cam = (R @ pts.T + t).T  # (N, 3)
    z = cam[:, 2]
    valid = z > 1e-6
    inv_z = np.zeros_like(z)
    inv_z[valid] = 1.0 / z[valid]
    x = cam[:, 0] * inv_z
    y = cam[:, 1] * inv_z
    u = K[0, 0] * x + K[0, 2]
    v = K[1, 1] * y + K[1, 2]
    return u, v, valid


def unproject_depth_to_world(depth, K, R, t, num_points, sample_mask=None, rng=None):
    """Convert a depth map + camera params to (M, 3) world points, M <= num_points."""
    if rng is None:
        rng = np.random.default_rng()
    d = np.asarray(depth, dtype=np.float32)
    if d.ndim != 2:
        return np.zeros((0, 3), dtype=np.float32)

    valid = np.isfinite(d) & (d > 1e-6)
    if sample_mask is not None:
        mask = np.asarray(sample_mask, dtype=bool)
        if mask.shape != d.shape:
            return np.zeros((0, 3), dtype=np.float32)
        valid &= mask

    ys, xs = np.where(valid)
    if ys.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    idx = rng.choice(ys.size, size=min(num_points, ys.size), replace=False)
    y = ys[idx].astype(np.float32)
    x = xs[idx].astype(np.float32)
    z = d[ys[idx], xs[idx]].astype(np.float32)

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    cam_pts = np.stack([(x - cx) * z / fx, (y - cy) * z / fy, z], axis=0)
    # cam = R * world + t  =>  world = R^T * (cam - t)
    world = (R.T @ (cam_pts - t)).T
    return world.astype(np.float32, copy=False)


def clean_depth(depth, clip_percentile):
    """Clip extreme depth values."""
    d = np.asarray(depth, dtype=np.float32)
    d = np.where(np.isfinite(d), d, 0.0)
    d[d <= 0] = 0.0
    if clip_percentile is not None and clip_percentile > 0 and np.any(d > 0):
        hi = float(np.percentile(d[d > 0], clip_percentile))
        d = np.clip(d, 0.0, hi)
    return d


def dilate_mask(mask, radius_px):
    """Binary dilation via cv2.  mask: (H, W) bool."""
    if radius_px <= 0:
        return mask
    k = 2 * radius_px + 1
    kernel = np.ones((k, k), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0

# ---------------------------------------------------------------------------
# COLMAP extraction (2026 format)
# ---------------------------------------------------------------------------

def extract_colmap_points_2026(sample):
    """Extract (N, 3) float32 COLMAP world points from a 2026-format sample.

    sample['colmap'] must be a ZIP archive containing points3D.txt.
    Fails fast if that file is missing (it is always present in the 2026 format).
    """
    colmap_blob = sample.get("colmap")
    if colmap_blob is None:
        return np.zeros((0, 3), dtype=np.float32)
    if not isinstance(colmap_blob, (bytes, bytearray, memoryview)):
        return np.zeros((0, 3), dtype=np.float32)

    try:
        with zipfile.ZipFile(BytesIO(colmap_blob)) as zf:
            if "points3D.txt" not in set(zf.namelist()):
                raise FileNotFoundError(
                    "COLMAP ZIP is missing points3D.txt -- "
                    "this is required in the 2026 dataset format")
            with zf.open("points3D.txt") as f:
                text = f.read().decode("utf-8", errors="ignore")
            # Format: POINT3D_ID X Y Z R G B ERROR TRACK[]
            # Filter comment/blank lines, parse columns 1-3 (X,Y,Z)
            from io import StringIO
            clean = "\n".join(l for l in text.split("\n") if l and not l.startswith("#"))
            if not clean:
                return np.zeros((0, 3), dtype=np.float32)
            return np.loadtxt(StringIO(clean), dtype=np.float32, usecols=(1, 2, 3))
    except zipfile.BadZipFile:
        pass
    return np.zeros((0, 3), dtype=np.float32)

# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def _codes_from_image(img):
    """Convert a PIL Image or numpy array to a (H, W) uint32 packed-RGB map."""
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return _pack_rgb_u32(arr)


def _row_majority(values):
    """Row-wise majority vote on (P, V) int array; -1 means "no vote".
    Returns (P,) with the most frequent non-negative value per row, or -1.

    Masks -1 entries before voting so that abstentions don't outvote
    actual labels (which happens when a point is visible in only 1-2 views).
    """
    P, V = values.shape
    result = np.full(P, -1, dtype=values.dtype)

    # For each row, find the most frequent non-negative value.
    # Vectorized approach: flatten valid entries per row using argmax on counts.
    # Since values are typically small non-negative ints (0-200), we can use
    # a simple max-of-first-valid approach for speed when V is small.
    for vi in range(V):
        # For rows still unset, take the first valid vote
        col = values[:, vi]
        unset = result == -1
        has_val = col >= 0
        update = unset & has_val
        result[update] = col[update]

    # Now refine: if a row has multiple different valid votes, pick the mode.
    # Check if any row has conflicting votes across views.
    has_any = np.any(values >= 0, axis=1)
    n_valid = np.sum(values >= 0, axis=1)
    needs_vote = has_any & (n_valid > 1)

    if np.any(needs_vote):
        for i in np.where(needs_vote)[0]:
            valid = values[i][values[i] >= 0]
            # Use numpy bincount for speed (values are small non-neg ints)
            counts = np.bincount(valid.astype(np.intp))
            result[i] = counts.argmax()

    return result

# ---------------------------------------------------------------------------
# Semantic fusion: house-centric, occluder-aware
# ---------------------------------------------------------------------------

def _fuse_labels_for_points(
    points_world, Ks, Rs, ts, ade_images, gestalt_images,
    ade_transparent_codes, ade_occluder_allowed_ids,
    min_support_views, valid_view_mask=None,
):
    """Multi-view semantic label fusion with majority voting.

    For each 3D point, project into every valid view:
      - ADE "envelope" class -> use the Gestalt label behind it.
      - ADE non-envelope -> keep if on the occluder allowlist.
    Then majority-vote across views.

    Returns dict: keep, visible_src, visible_id, behind_gest_id, support
    """
    P = points_world.shape[0]
    V = min(len(Ks), len(Rs), len(ts), len(ade_images), len(gestalt_images))
    empty = {
        "keep": np.zeros(P, dtype=bool),
        "visible_src": np.zeros(P, np.uint8),
        "visible_id": np.full(P, -1, np.int16),
        "behind_gest_id": np.full(P, -1, np.int16),
        "support": np.zeros(P, np.uint8),
    }
    if P == 0 or V == 0:
        return empty

    # Per-view labels. src: 1=gestalt, 2=ade; -1 = no contribution.
    visible_src_pv = np.full((P, V), -1, dtype=np.int8)
    visible_id_pv = np.full((P, V), -1, dtype=np.int32)
    behind_id_pv = np.full((P, V), -1, dtype=np.int32)
    support = np.zeros(P, dtype=np.int32)

    ade_allowed_set = set(ade_occluder_allowed_ids.tolist())
    ade_transparent_u32 = ade_transparent_codes.astype(np.uint32, copy=False)
    gest_invalid_arr = np.array(list(GEST_INVALID_CODES), dtype=np.uint32)

    for vi in range(V):
        if valid_view_mask is not None and not valid_view_mask[vi]:
            continue

        K = np.asarray(Ks[vi], np.float32)
        R = np.asarray(Rs[vi], np.float32)
        t = np.asarray(ts[vi], np.float32).reshape(3, 1)

        ade_codes_img = _codes_from_image(ade_images[vi])
        gest_codes_img = _codes_from_image(gestalt_images[vi])
        H, W = ade_codes_img.shape

        u, v, valid = project_world_points(points_world, K, R, t)
        in_img = valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not np.any(in_img):
            continue

        ui = np.clip(np.round(u[in_img]).astype(np.int32), 0, W - 1)
        vi_pix = np.clip(np.round(v[in_img]).astype(np.int32), 0, H - 1)
        ade_codes = ade_codes_img[vi_pix, ui]
        gest_codes = gest_codes_img[vi_pix, ui]

        in_house = ~np.isin(gest_codes, gest_invalid_arr)
        if not np.any(in_house):
            continue

        idx = np.where(in_img)[0][in_house]
        ade_codes_h = ade_codes[in_house]
        gest_codes_h = gest_codes[in_house]

        behind_local = np.array(
            [GEST_RGBCODE_TO_ID.get(int(c), -1) for c in gest_codes_h],
            dtype=np.int32)
        behind_id_pv[idx, vi] = behind_local

        ade_is_transparent = np.isin(ade_codes_h, ade_transparent_u32)

        # Case A: ADE is envelope -- use Gestalt label.
        mask_a = ade_is_transparent & (behind_local >= 0)
        if np.any(mask_a):
            visible_src_pv[idx[mask_a], vi] = 1
            visible_id_pv[idx[mask_a], vi] = behind_local[mask_a]

        # Case B: ADE is non-envelope -- use ADE label (allowlist-filtered).
        mask_b = ~ade_is_transparent
        if np.any(mask_b):
            ade_local = np.array(
                [ADE_RGBCODE_TO_ID.get(int(c), -1) for c in ade_codes_h[mask_b]],
                dtype=np.int32)
            on_allowlist = np.array(
                [int(a) in ade_allowed_set for a in ade_local], dtype=bool
            ) & (ade_local >= 0)
            if np.any(on_allowlist):
                visible_src_pv[idx[mask_b][on_allowlist], vi] = 2
                visible_id_pv[idx[mask_b][on_allowlist], vi] = ade_local[on_allowlist]

        support[idx] += 1

    # ---- Aggregate across views via majority vote ----
    keep = (support >= min_support_views) & np.any(visible_src_pv >= 0, axis=1)

    # Combine (src, id) into a single key for voting, then split back.
    # src in {1,2} and id in [0, ~150], so stride=100k avoids collisions.
    VIS_STRIDE = 100_000
    vis_key = np.where(
        visible_src_pv >= 0,
        visible_src_pv.astype(np.int64) * VIS_STRIDE + visible_id_pv.astype(np.int64),
        -1)
    voted_key = _row_majority(vis_key)
    voted_behind = _row_majority(behind_id_pv)

    final_src = np.zeros(P, dtype=np.uint8)
    final_id = np.full(P, -1, dtype=np.int16)
    ok = voted_key >= 0
    if np.any(ok):
        final_src[ok] = (voted_key[ok] // VIS_STRIDE).astype(np.uint8)
        final_id[ok] = (voted_key[ok] % VIS_STRIDE).astype(np.int16)

    # ---- Vote confidence metadata ----
    n_views_voted = np.sum(visible_src_pv >= 0, axis=1).astype(np.uint8)

    # Fraction of voting views that agreed with the majority label
    vote_frac = np.zeros(P, dtype=np.float32)
    if np.any(ok):
        for i in np.where(ok)[0]:
            votes = vis_key[i][vis_key[i] >= 0]
            if len(votes) > 0:
                vote_frac[i] = (votes == voted_key[i]).sum() / len(votes)

    return {
        "keep": keep,
        "visible_src": final_src,
        "visible_id": final_id,
        "behind_gest_id": voted_behind.astype(np.int16),
        "support": support.astype(np.uint8),
        "n_views_voted": n_views_voted,
        "vote_frac": vote_frac,
    }

# ---------------------------------------------------------------------------
# Compact scene builder (2026 dataset format)
# ---------------------------------------------------------------------------

def _resolve_ade_codes(cfg):
    """Return (transparent_codes, occluder_ids) for the given config.
    Uses precomputed module-level arrays when the config has default names.
    """
    if cfg.ade_transparent_classes == ADE_TRANSPARENT_NAMES:
        transparent = _DEFAULT_ADE_TRANSPARENT_CODES
    else:
        transparent = np.array(
            [c for n in cfg.ade_transparent_classes
             if (c := _name_to_packed_rgb(n, ade20k_color_mapping)) is not None],
            dtype=np.uint32)

    if cfg.ade_occluder_allowlist == ADE_OCCLUDER_ALLOWLIST_NAMES:
        occluder_ids = _DEFAULT_ADE_OCCLUDER_IDS
    else:
        occluder_ids = np.array(
            sorted({ADE_RGBCODE_TO_ID[c]
                    for n in cfg.ade_occluder_allowlist
                    if (c := _name_to_packed_rgb(n, ade20k_color_mapping)) is not None
                    and c in ADE_RGBCODE_TO_ID}),
            dtype=np.int32)
    return transparent, occluder_ids


def _parse_gt_array(sample, key, dtype, expected_cols):
    """Parse an optional ground-truth array from the sample dict."""
    raw = sample.get(key)
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=dtype)
    if arr.ndim == 2 and arr.shape[1] == expected_cols:
        return arr
    return None


def build_compact_scene(sample, cfg, rng):
    """Build a compact semantic point representation from a HuggingFace sample.

    Expected sample keys: K, R, t, ade, gestalt, depth, colmap,
    pose_only_in_colmap, wf_vertices (opt), wf_edges (opt), __key__ (opt).

    Returns dict (xyz, source, visible_src, visible_id, behind_gest_id,
    gt_vertices, gt_edges, sample_id) or None if no points survive fusion.
    """
    Ks = sample.get("K") or []
    Rs = sample.get("R") or []
    ts = sample.get("t") or []
    ade_imgs = sample.get("ade") or []
    gest_imgs = sample.get("gestalt") or []
    depths = sample.get("depth") or []
    pose_flags = sample.get("pose_only_in_colmap") or []

    V = min(len(Ks), len(Rs), len(ts), len(ade_imgs), len(gest_imgs))
    if V == 0:
        return None

    valid_view = [not (vi < len(pose_flags) and pose_flags[vi]) for vi in range(V)]
    if not any(valid_view):
        return None

    # ---- COLMAP points ----
    colmap_pts = extract_colmap_points_2026(sample)

    # ---- Precompute house masks (from Gestalt), optionally dilated ----
    gest_invalid_arr = np.array(list(GEST_INVALID_CODES), dtype=np.uint32)
    house_masks = []
    for vi in range(V):
        if not valid_view[vi]:
            house_masks.append(None)
            continue
        mask = ~np.isin(_codes_from_image(gest_imgs[vi]), gest_invalid_arr)
        if cfg.house_mask_dilate_px > 0:
            mask = dilate_mask(mask, cfg.house_mask_dilate_px)
        house_masks.append(mask)

    # ---- Sample depth points per view ----
    depth_points_all = []
    for vi in range(min(V, len(depths))):
        if not valid_view[vi] or depths[vi] is None:
            continue
        d = clean_depth(
            np.asarray(depths[vi], dtype=np.float32) * cfg.depth_scale,
            cfg.depth_clip_percentile)
        pts = unproject_depth_to_world(
            depth=d,
            K=np.asarray(Ks[vi], np.float32),
            R=np.asarray(Rs[vi], np.float32),
            t=np.asarray(ts[vi], np.float32).reshape(3, 1),
            num_points=cfg.depth_points_per_view,
            sample_mask=house_masks[vi], rng=rng)
        if pts.shape[0]:
            depth_points_all.append(pts)

    # ---- Combine COLMAP + depth points ----
    pts_list, src_list = [], []
    if colmap_pts.shape[0]:
        pts_list.append(colmap_pts)
        src_list.append(np.zeros(colmap_pts.shape[0], dtype=np.uint8))   # 0=colmap
    if depth_points_all:
        all_depth = np.concatenate(depth_points_all, axis=0)
        pts_list.append(all_depth)
        src_list.append(np.ones(all_depth.shape[0], dtype=np.uint8))     # 1=depth
    if not pts_list:
        return None

    points_world = np.concatenate(pts_list, axis=0).astype(np.float32, copy=False)
    point_source = np.concatenate(src_list, axis=0).astype(np.uint8, copy=False)

    # ---- Fuse semantic labels ----
    ade_transparent_arr, ade_allow_ids = _resolve_ade_codes(cfg)
    fused = _fuse_labels_for_points(
        points_world=points_world, Ks=Ks, Rs=Rs, ts=ts,
        ade_images=ade_imgs, gestalt_images=gest_imgs,
        ade_transparent_codes=ade_transparent_arr,
        ade_occluder_allowed_ids=ade_allow_ids,
        min_support_views=cfg.min_support_views,
        valid_view_mask=valid_view)

    keep = fused["keep"]
    if not np.any(keep):
        return None

    return {
        "xyz": points_world[keep],
        "source": point_source[keep],               # 0=colmap, 1=monodepth
        "visible_src": fused["visible_src"][keep],   # 1=gestalt, 2=ade
        "visible_id": fused["visible_id"][keep],
        "behind_gest_id": fused["behind_gest_id"][keep],
        "n_views_voted": fused["n_views_voted"][keep],
        "vote_frac": fused["vote_frac"][keep],
        "gt_vertices": _parse_gt_array(sample, "wf_vertices", np.float32, 3),
        "gt_edges": _parse_gt_array(sample, "wf_edges", np.int64, 2),
        "sample_id": sample.get("__key__", None),
    }
