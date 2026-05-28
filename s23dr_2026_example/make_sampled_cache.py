#!/usr/bin/env python3
"""Stage 2: priority-sample cached .pt scenes into fixed-size .npz files.

Reads the per-scene .pt files produced by cache_scenes.py, priority-samples
a fixed number of points (2048 or 4096), normalizes, and writes one .npz per
scene (~50KB at 2048, ~100KB at 4096).

A fixed seed is used so every scene gets one deterministic sample -- no
per-epoch sampling augmentation, every epoch sees the same points.

Usage:
  python -m s23dr_2026_example.make_sampled_cache \\
      --in-dir cache/full --out-dir cache/sampled_2048 --seq-len 2048
  python -m s23dr_2026_example.make_sampled_cache \\
      --in-dir cache/full --out-dir cache/sampled_4096 --seq-len 4096

The 3:1 colmap:depth quota ratio is fixed: at seq_len=2048 that's
colmap=1536/depth=512; at seq_len=4096 that's colmap=3072/depth=1024.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch


# Priority sampling (same logic as train.py)
def _priority_sample(source, group_id, seq_len, colmap_quota, depth_quota):
    def pick(src_id, quota):
        base = source == src_id
        picked, remaining = [], quota
        for tier in range(5):
            if remaining <= 0:
                break
            pool = np.where(base & (group_id == tier))[0]
            if len(pool) == 0:
                continue
            np.random.shuffle(pool)
            take = min(remaining, len(pool))
            picked.append(pool[:take])
            remaining -= take
        if remaining > 0:
            pool = np.where(base & (group_id >= 0))[0]
            if len(pool) > 0:
                np.random.shuffle(pool)
                picked.append(pool[:min(remaining, len(pool))])
                remaining -= min(remaining, len(pool))
        return np.concatenate(picked) if picked else np.array([], dtype=np.int64), remaining

    idx_c, rem_c = pick(0, colmap_quota)
    idx_d, rem_d = pick(1, depth_quota)

    if rem_c > 0:
        extra = np.setdiff1d(np.where((source == 1) & (group_id >= 0))[0], idx_d)
        np.random.shuffle(extra)
        idx_d = np.concatenate([idx_d, extra[:rem_c]])
    if rem_d > 0:
        extra = np.setdiff1d(np.where((source == 0) & (group_id >= 0))[0], idx_c)
        np.random.shuffle(extra)
        idx_c = np.concatenate([idx_c, extra[:rem_d]])

    indices = np.concatenate([idx_c, idx_d])
    num_valid = len(indices)
    if num_valid < seq_len:
        if num_valid == 0:
            return np.zeros(seq_len, dtype=np.int64), np.zeros(seq_len, dtype=bool)
        indices = np.concatenate([indices, np.full(seq_len - num_valid, indices[-1])])
    mask = np.zeros(seq_len, dtype=bool)
    mask[:num_valid] = True
    return indices[:seq_len], mask


def _process_sample(d, seq_len, colmap_q, depth_q):
    """Sample and normalize one cached scene dict into a small npz-ready dict."""
    xyz = np.asarray(d["xyz"], np.float32)
    source = np.asarray(d["source"], np.uint8)
    group_id = np.asarray(d["group_id"], np.int8)
    class_id = np.asarray(d["class_id"], np.uint8)
    vis_src = np.asarray(d["visible_src"], np.uint8)
    vis_id = np.asarray(d["visible_id"], np.int16)
    center = np.asarray(d["center"], np.float32)
    scale = float(d["scale"])
    gt_v = np.asarray(d["gt_vertices"], np.float32)
    gt_e = np.asarray(d["gt_edges"], np.int32)

    indices, mask = _priority_sample(source, group_id, seq_len, colmap_q, depth_q)
    xyz_norm = ((xyz[indices] - center) / scale).astype(np.float32)
    gt_seg = np.stack([gt_v[gt_e[:, 0]], gt_v[gt_e[:, 1]]], axis=1)
    gt_seg_norm = ((gt_seg - center) / scale).astype(np.float32)

    result = {
        "xyz_norm": xyz_norm,
        "class_id": class_id[indices].astype(np.uint8),
        "source": source[indices].astype(np.uint8),
        "mask": mask,
        "gt_segments": gt_seg_norm,
        "scale": np.float32(scale),
        "center": center,
        "gt_vertices": gt_v,
        "gt_edges": gt_e,
        "visible_src": vis_src[indices].astype(np.uint8),
        "visible_id": vis_id[indices].astype(np.int16),
    }
    if "behind_gest_id" in d:
        result["behind"] = np.asarray(d["behind_gest_id"], np.int16)[indices]
    if "n_views_voted" in d:
        result["n_views_voted"] = np.asarray(d["n_views_voted"], np.uint8)[indices]
    if "vote_frac" in d:
        result["vote_frac"] = np.asarray(d["vote_frac"], np.float32)[indices]
    if "gt_edge_classes" in d:
        result["gt_edge_classes"] = np.asarray(d["gt_edge_classes"], np.int64)
    return result


def main():
    p = argparse.ArgumentParser(description="Stage 2: cached .pt -> sampled .npz")
    p.add_argument("--in-dir", required=True, help="Directory of .pt files from cache_scenes.py")
    p.add_argument("--out-dir", required=True, help="Output directory for .npz files")
    p.add_argument("--seq-len", type=int, default=2048, help="Points per sample (2048 or 4096)")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    colmap_q = args.seq_len * 3 // 4
    depth_q = args.seq_len - colmap_q
    print(f"seq_len={args.seq_len}  colmap={colmap_q}  depth={depth_q}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)

    files = sorted(Path(args.in_dir).glob("*.pt"))
    print(f"Found {len(files)} .pt files in {args.in_dir}")

    done = 0
    t0 = time.perf_counter()
    for f in files:
        out_f = out_dir / (f.stem + ".npz")
        if out_f.exists():
            done += 1
            continue
        d = torch.load(f, weights_only=False)
        result = _process_sample(d, args.seq_len, colmap_q, depth_q)
        np.savez(out_f, **result)
        done += 1
        if done % 2000 == 0:
            rate = done / (time.perf_counter() - t0)
            print(f"  {done}/{len(files)} [{rate:.0f}/s]")

    elapsed = time.perf_counter() - t0
    print(f"Done. {done} files in {elapsed:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()


