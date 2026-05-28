"""S23DR 2026 submission: learned wireframe prediction from fused point clouds.

Pipeline: raw sample -> point fusion -> priority sample 2048 -> model -> post-process -> wireframe
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import subprocess
import sys

def install_if_missing(package):
    try:
        __import__(package.split("==")[0])
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_if_missing("scipy")
install_if_missing("pandas")

from pathlib import Path
from tqdm import tqdm
import json
import sys
import time

import numpy as np
import torch


def empty_solution():
    return np.zeros((2, 3)), [(0, 1)]


# ---------------------------------------------------------------------------
# Point fusion + sampling (from cache_scenes.py / make_sampled_cache.py)
# ---------------------------------------------------------------------------

# Add our package to path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from s23dr_2026_example.point_fusion import build_compact_scene, FuserConfig
from s23dr_2026_example.cache_scenes import (
    _compute_group_and_class, _compute_smart_center_scale,
)
from s23dr_2026_example.make_sampled_cache import _priority_sample

# Tokenizer / model imports
from s23dr_2026_example.tokenizer import EdgeDepthSequenceConfig
from s23dr_2026_example.model import EdgeDepthSegmentsModel
from s23dr_2026_example.segment_postprocess import merge_vertices_iterative
from s23dr_2026_example.varifold import segments_to_vertices_edges
from s23dr_2026_example.postprocess_v2 import snap_to_point_cloud, snap_horizontal

SEQ_LEN = 4096
COLMAP_QUOTA = 3072
DEPTH_QUOTA = 1024
CONF_THRESH = 0.4
MERGE_THRESH = 0.4
SNAP_RADIUS = 0.5

# Test-time augmentation: 3 priority-sample seeds + Hungarian matching.
# Local 100-sample A/B was +0.003 mean / +0.029 q5 vs single-pass, BUT the
# 3x inference cost (commit b6bc99a) timed out on HF Space and produced
# all-zero scores. Disabled. Re-enable only with single-seed (2x not 3x)
# or after profiling shows the HF Space can fit 3x within its time limit.
USE_TTA = False
TTA_SEEDS = (2718, 31415, 42)
TTA_MIN_PASSES = 2

# Multi-checkpoint ensemble: tried 2026-05-23 with checkpoint2.pt fine-tuned
# under heavy aug (jitter 0.01 + drop 0.1) for 20k steps from checkpoint.pt.
# Local 50-sample A/B (new model alone vs original alone): mean dropped by
# 0.040 with 20 big losses vs 5 big wins -- the aug degraded model accuracy.
# Ensemble of [original + new] therefore can't improve over original alone.
# Keep flag disabled; need a more careful training run before re-enabling.
USE_ENSEMBLE = False
ENSEMBLE_MIN_PASSES = 1
TTA_PLUS_ENSEMBLE_MIN_PASSES = 2


def fuse_and_sample(sample, cfg, rng):
    """Run point fusion + priority sampling on a raw dataset sample.

    Returns a dict with xyz_norm, class_id, source, mask, center, scale, etc.
    ready for model inference. Returns None if fusion fails.
    """
    try:
        scene = build_compact_scene(sample, cfg, rng)
    except Exception as e:
        print(f"  Fusion failed: {e}")
        return None

    xyz = scene["xyz"]
    source = scene["source"]

    if len(xyz) < 10:
        return None

    # Compute group_id and class_id (same as cache_scenes.py)
    behind_id = scene.get("behind_gest_id", np.full(len(xyz), -1, dtype=np.int16))
    group_id, class_id = _compute_group_and_class(
        scene["visible_src"], scene["visible_id"], behind_id, source)

    # Normalize
    center, scale = _compute_smart_center_scale(xyz, source)

    # Priority sample
    indices, mask = _priority_sample(source, group_id, SEQ_LEN, COLMAP_QUOTA, DEPTH_QUOTA)

    xyz_norm = (xyz[indices] - center) / scale

    result = {
        "xyz_norm": xyz_norm.astype(np.float32),
        "class_id": class_id[indices].astype(np.int64),
        "source": source[indices].astype(np.int64),
        "mask": mask,
        "center": center.astype(np.float32),
        "scale": np.float32(scale),
    }

    # Optional fields
    if "behind_gest_id" in scene:
        behind = np.clip(scene["behind_gest_id"][indices].astype(np.int16), 0, None)
        result["behind"] = behind.astype(np.int64)
    if "n_views_voted" in scene:
        result["n_views_voted"] = scene["n_views_voted"][indices].astype(np.float32)
    if "vote_frac" in scene:
        result["vote_frac"] = scene["vote_frac"][indices].astype(np.float32)

    # Visible src/id for snap post-processing
    result["visible_src"] = scene["visible_src"][indices].astype(np.int64)
    result["visible_id"] = scene["visible_id"][indices].astype(np.int64)

    return result


def load_model(checkpoint_path, device):
    """Load model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})

    norm_class = torch.nn.RMSNorm if args.get("rms_norm") else None
    seq_cfg = EdgeDepthSequenceConfig(
        seq_len=SEQ_LEN, colmap_points=COLMAP_QUOTA, depth_points=DEPTH_QUOTA)

    model = EdgeDepthSegmentsModel(
        seq_cfg=seq_cfg,
        segments=args.get("segments", 64),
        hidden=args.get("hidden", 256),
        num_heads=args.get("num_heads", 4),
        kv_heads_cross=args.get("kv_heads_cross", 2),
        kv_heads_self=args.get("kv_heads_self", 2),
        dim_feedforward=args.get("ff", 1024),
        dropout=args.get("dropout", 0.1),
        latent_tokens=args.get("latent_tokens", 256),
        latent_layers=args.get("latent_layers", 7),
        decoder_layers=args.get("decoder_layers", 3),
        cross_attn_interval=args.get("cross_attn_interval", 4),
        norm_class=norm_class,
        activation=args.get("activation", "gelu"),
        segment_conf=args.get("segment_conf", True),
        behind_emb_dim=args.get("behind_emb_dim", 8),
        use_vote_features=args.get("vote_features", True),
        arch=args.get("arch", "perceiver"),
        encoder_layers=args.get("encoder_layers", 4),
        pre_encoder_layers=args.get("pre_encoder_layers", 0),
        segment_param=args.get("segment_param", "midpoint_dir_len"),
        qk_norm=args.get("qk_norm", True),
    ).to(device)

    # Handle torch.compile _orig_mod prefix
    state = ckpt["model"]
    fixed = {k.replace("segmenter._orig_mod.", "segmenter."): v
             for k, v in state.items()}
    model.load_state_dict(fixed, strict=True)
    model.eval()
    return model


def build_tokens_single(sample_dict, model, device):
    """Build token tensor for a single sample (no DataLoader)."""
    xyz = torch.as_tensor(sample_dict["xyz_norm"], dtype=torch.float32).unsqueeze(0).to(device)
    cid = torch.as_tensor(sample_dict["class_id"], dtype=torch.long).unsqueeze(0).to(device)
    src = torch.as_tensor(sample_dict["source"], dtype=torch.long).unsqueeze(0).to(device)
    masks = torch.as_tensor(sample_dict["mask"], dtype=torch.bool).unsqueeze(0).to(device)

    B, T, _ = xyz.shape
    tok = model.tokenizer
    fourier = tok.pos_enc(xyz.reshape(-1, 3)).reshape(B, T, -1) \
        if tok.pos_enc is not None else xyz.new_zeros(B, T, 0)
    parts = [xyz, fourier, tok.label_emb(cid), tok.src_emb(src.clamp(0, 1))]

    if tok.behind_emb_dim > 0:
        if "behind" in sample_dict:
            beh = torch.as_tensor(sample_dict["behind"], dtype=torch.long).unsqueeze(0).to(device)
        else:
            beh = xyz.new_zeros(B, T, dtype=torch.long)
        parts.append(tok.behind_emb(beh))

    if tok.use_vote_features:
        if "n_views_voted" in sample_dict and "vote_frac" in sample_dict:
            nv = ((torch.as_tensor(sample_dict["n_views_voted"], dtype=torch.float32).unsqueeze(0).to(device) - 2.7) / 1.0).unsqueeze(-1)
            vf = ((torch.as_tensor(sample_dict["vote_frac"], dtype=torch.float32).unsqueeze(0).to(device) - 0.5) / 0.25).unsqueeze(-1)
            parts.extend([nv, vf])
        else:
            parts.extend([xyz.new_zeros(B, T, 1), xyz.new_zeros(B, T, 1)])

    tokens = torch.cat(parts, dim=-1)
    return tokens, masks


def predict_sample(sample_dict, model, device):
    """Run model inference + post-processing on a fused sample.

    Returns (vertices, edges) in world space.
    """
    tokens, masks = build_tokens_single(sample_dict, model, device)
    scale = float(sample_dict["scale"])
    center = sample_dict["center"]

    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16,
                                          enabled=(device.type == 'cuda')):
        out = model.forward_tokens(tokens, masks)

    segs = out["segments"][0].float().cpu()
    conf = torch.sigmoid(out["conf"][0].float()).cpu().numpy() if "conf" in out else None

    # Confidence filter
    if conf is not None:
        keep = conf > CONF_THRESH
        segs = segs[keep]
    if len(segs) < 1:
        return empty_solution()

    # To world space
    segs_world = segs.numpy() * scale + center

    # Vertices + edges from segments
    pv, pe = segments_to_vertices_edges(torch.tensor(segs_world))
    pv, pe = pv.numpy(), np.array(pe, dtype=np.int32)

    # Merge
    pv, pe = merge_vertices_iterative(pv, pe)

    # Snap to point cloud. Target classes: apex (0), eave_end_point (1),
    # flashing_end_point (2) — the three semantic POINT classes. Local 100-sample
    # A/B vs original [1,2] showed +0.005 hss_mean (10 big wins vs 8 regressions).
    # NB: the reverted commit 6cf3fbd tried [1,2,3] but class 3 is rake (an edge
    # class, not a point class) — that was the bug. [0,1,2] is the correct fix.
    xyz_norm = sample_dict["xyz_norm"]
    mask = sample_dict["mask"]
    cid = sample_dict["class_id"]
    xyz_world = xyz_norm[mask] * scale + center
    cid_valid = cid[mask]
    pv = snap_to_point_cloud(
        pv, xyz_world, cid_valid, snap_radius=SNAP_RADIUS, target_classes=[0, 1, 2])

    # Horizontal snap
    pv = snap_horizontal(pv, pe)

    if len(pv) < 2 or len(pe) < 1:
        return empty_solution()

    edges = [(int(a), int(b)) for a, b in pe]
    return pv, edges

def hybrid_merge(pred_v, pred_e, track_v, track_e, merge_radius=0.8):
    if len(track_v) == 0:
        return pred_v, pred_e
        
    pred_v = np.array(pred_v) if isinstance(pred_v, list) else pred_v
    track_v = np.array(track_v)
    
    # Filter out NaNs and Infs from track_v
    valid_mask = np.isfinite(track_v).all(axis=1)
    if not valid_mask.all():
        valid_indices = np.where(valid_mask)[0]
        idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_indices)}
        track_v = track_v[valid_mask]
        new_track_e = []
        for u, v in track_e:
            if u in idx_map and v in idx_map:
                new_track_e.append((idx_map[u], idx_map[v]))
        track_e = new_track_e
        
    if len(track_v) == 0:
        return pred_v, pred_e
    
    # We will append track vertices that are NOT close to any pred_v
    if len(pred_v) > 0:
        from scipy.spatial import cKDTree
        tree = cKDTree(pred_v)
        dists, indices = tree.query(track_v, k=1)
    else:
        dists = np.full(len(track_v), np.inf)
        indices = np.zeros(len(track_v), dtype=int)
        
    # Map track vertex indices to final vertex indices
    track_to_final = {}
    new_vertices = []
    
    for i, (d, idx) in enumerate(zip(dists, indices)):
        if d <= merge_radius and len(pred_v) > 0:
            # Map to existing pred_v
            track_to_final[i] = int(idx)
        else:
            # Add as new vertex
            track_to_final[i] = len(pred_v) + len(new_vertices)
            new_vertices.append(track_v[i])
            
    final_v = list(pred_v) + new_vertices
    final_e = list(pred_e)
    
    # Add track edges, mapping their indices
    existing_edges = set()
    for u, v in final_e:
        existing_edges.add((min(u, v), max(u, v)))
        
    for u_t, v_t in track_e:
        u_f = track_to_final.get(u_t)
        v_f = track_to_final.get(v_t)
        if u_f is not None and v_f is not None and u_f != v_f:
            e = (min(u_f, v_f), max(u_f, v_f))
            if e not in existing_edges:
                # ONLY append the tracked edge if it connects to a NEWLY DISCOVERED vertex.
                # This prevents the geometric tracker from aggressively re-wiring the learned model's existing topology!
                if u_f >= len(pred_v) or v_f >= len(pred_v):
                    final_e.append(e)
                    existing_edges.add(e)
                    
    return np.array(final_v), final_e

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t_start = time.time()

    # Load params
    param_path = Path("params.json")
    with param_path.open() as f:
        params = json.load(f)
    print(f"Competition: {params.get('competition_id', '?')}")
    print(f"Dataset: {params.get('dataset', '?')}")

    # Load test data
    data_path = Path("/tmp/data")
    if not data_path.exists():
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=params["dataset"],
            local_dir="/tmp/data",
            repo_type="dataset",
        )

    from datasets import load_dataset
    data_files = {}
    public_tars = sorted([str(p) for p in data_path.rglob('*public*/**/*.tar')])
    private_tars = sorted([str(p) for p in data_path.rglob('*private*/**/*.tar')])
    if public_tars:
        data_files["validation"] = public_tars
    if private_tars:
        data_files["test"] = private_tars
    print(f"Data files: {data_files}")
    loading_scripts = sorted(data_path.rglob('*.py'))
    loading_script = str(loading_scripts[0]) if loading_scripts else str(data_path)

    dataset = load_dataset(
        loading_script,
        data_files=data_files,
        trust_remote_code=True,
        writer_batch_size=100,
    )
    print(f"Loaded: {dataset}")

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    checkpoint_path = SCRIPT_DIR / "checkpoint.pt"
    
    # Auto-download checkpoint if missing or just an LFS pointer
    if not checkpoint_path.exists() or checkpoint_path.stat().st_size < 1000:
        print("Downloading checkpoint.pt from upstream learned baseline...")
        import urllib.request
        ckpt_url = "https://huggingface.co/jacklangerman/s23dr-2026-submission/resolve/main/checkpoint.pt"
        urllib.request.urlretrieve(ckpt_url, str(checkpoint_path))
        print("Downloaded checkpoint.pt")
        
    model = load_model(checkpoint_path, device)
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # Vertex position regressor (DINOv2 patch features per vertex → 3D offset).
    # Trained on 800 samples. Predicts a small move toward the nearest GT vertex
    # using bilinear-sampled DINOv2 features and per-vertex geometric features.
    # Local 200-sample A/B vs edge-classifier-only: +0.0065 hss_mean (t=+2.39,
    # 109 wins / 79 losses). Combined vs baseline: +0.0095 (t=+3.84).
    # Offset clamped at 0.3m magnitude for safety.
    vertex_regressor_bundle = None
    vertex_regressor_path = SCRIPT_DIR / "vertex_regressor_v4_800.pt"
    if vertex_regressor_path.exists():
        try:
            from vertex_regressor_v4 import load_regressor_v4
            from edge_classifier_v4 import get_dino_model as _get_dino
            vr_model, vrg_mean, vrg_std, vre_mean, vre_std = load_regressor_v4(
                str(vertex_regressor_path), device="cpu")
            dino_vr = _get_dino(device=device)
            vertex_regressor_bundle = {
                "model": vr_model,
                "dino": dino_vr,
                "dino_device": device,
                "mean": vrg_mean.cpu().numpy() if hasattr(vrg_mean, "cpu") else vrg_mean,
                "std": vrg_std.cpu().numpy() if hasattr(vrg_std, "cpu") else vrg_std,
                "edge_feat_mean": vre_mean.cpu().numpy() if hasattr(vre_mean, "cpu") else vre_mean,
                "edge_feat_std": vre_std.cpu().numpy() if hasattr(vre_std, "cpu") else vre_std,
                "max_move_meters": 0.3,
            }
            print(f"Vertex regressor v4 loaded ({sum(p.numel() for p in vr_model.parameters()):,} head params)")
        except Exception as vr_err:
            print(f"Vertex regressor load failed: {vr_err}; running without")
            vertex_regressor_bundle = None
    else:
        print(f"No vertex_regressor_v4_800.pt at {vertex_regressor_path}; running without")

    # Edge classifier (DINOv2 patch features + geometric features → P(keep edge)).
    # Trained on 400 samples with match_radius=0.4. Best val acc 82.1%.
    # Operating point: thresh=0.15, min_keep=0.85 — drop only the bottom ~15%
    # of edges that the classifier is most confident are wrong. 200-sample local
    # A/B: +0.0030 hss_mean (t=+1.26, 100 wins / 77 losses).
    edge_classifier_bundle = None
    edge_classifier_path = SCRIPT_DIR / "edge_classifier_v4_400.pt"
    if edge_classifier_path.exists():
        try:
            from edge_classifier_v4 import load_classifier_v4, get_dino_model
            ec_model, g_mean, g_std, e_mean, e_std = load_classifier_v4(
                str(edge_classifier_path), device="cpu")
            dino = get_dino_model(device=device)
            edge_classifier_bundle = {
                "model": ec_model,
                "dino": dino,
                "dino_device": device,
                "mean": g_mean.cpu().numpy() if hasattr(g_mean, "cpu") else g_mean,
                "std": g_std.cpu().numpy() if hasattr(g_std, "cpu") else g_std,
                "edge_feat_mean": e_mean.cpu().numpy() if hasattr(e_mean, "cpu") else e_mean,
                "edge_feat_std": e_std.cpu().numpy() if hasattr(e_std, "cpu") else e_std,
                "threshold": 0.15,
                "min_keep_frac": 0.85,
            }
            print(f"Edge classifier v4 loaded ({sum(p.numel() for p in ec_model.parameters()):,} head params + frozen DINOv2)")
        except Exception as ec_err:
            print(f"Edge classifier load failed: {ec_err}; running without")
            edge_classifier_bundle = None
    else:
        print(f"No edge_classifier_v4_400.pt at {edge_classifier_path}; running without")

    # Optional: load 2nd checkpoint for ensemble inference
    ensemble_models = None
    if USE_ENSEMBLE:
        checkpoint2_path = SCRIPT_DIR / "checkpoint2.pt"
        if checkpoint2_path.exists() and checkpoint2_path.stat().st_size > 1000:
            model2 = load_model(checkpoint2_path, device)
            ensemble_models = [model, model2]
            print(f"Ensemble: loaded 2 models for cross-checkpoint averaging")
        else:
            print(f"USE_ENSEMBLE=True but checkpoint2.pt not present; running single-model")

    # Point fusion config
    cfg = FuserConfig()
    rng = np.random.RandomState(2718)

    # Process all samples
    solution = []
    total_samples = sum(len(dataset[s]) for s in dataset)
    processed = 0

    for subset_name in dataset:
        print(f"\nProcessing {subset_name} ({len(dataset[subset_name])} samples)...")

        for sample in tqdm(dataset[subset_name], desc=subset_name):
            order_id = sample["order_id"]

            # Diagnostic: input signal strength. No behavior change.
            n_colmap_pts = -1
            try:
                from hoho2025.example_solutions import convert_entry_to_human_readable
                _good = convert_entry_to_human_readable(sample)
                _rec = _good.get('colmap') or _good.get('colmap_binary')
                if _rec is not None:
                    n_colmap_pts = len(_rec.points3D)
            except Exception:
                pass

            track_v_count, track_e_count = 0, 0
            pred_status = "ok"
            n_fused_pts = 0

            if ensemble_models is not None and USE_TTA:
                # 2-model ensemble × multi-seed TTA: 2 * len(seeds) total passes.
                # Strict cross-pass agreement filters spurious segments.
                try:
                    from ensemble import predict_sample_ensemble
                    pred_v, pred_e = predict_sample_ensemble(
                        sample, cfg, ensemble_models, device,
                        seeds=TTA_SEEDS,
                        min_passes_for_keep=TTA_PLUS_ENSEMBLE_MIN_PASSES,
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as e:
                    import traceback
                    print(f"  Ensemble+TTA failed for {order_id}:\n{traceback.format_exc()}")
                    pred_v, pred_e = empty_solution()
                    pred_status = "ensemble_tta_failed"
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            elif ensemble_models is not None:
                # 2-model ensemble, single seed
                try:
                    from ensemble import predict_sample_ensemble
                    pred_v, pred_e = predict_sample_ensemble(
                        sample, cfg, ensemble_models, device,
                        seeds=(2718,),
                        min_passes_for_keep=ENSEMBLE_MIN_PASSES,
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as e:
                    import traceback
                    print(f"  Ensemble failed for {order_id}:\n{traceback.format_exc()}")
                    pred_v, pred_e = empty_solution()
                    pred_status = "ensemble_failed"
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            elif USE_TTA:
                # Multi-seed TTA: fuse + predict 3 times, Hungarian-match segments
                # across passes, drop those without min_passes agreement.
                try:
                    from tta import predict_sample_tta_hungarian
                    pred_v, pred_e = predict_sample_tta_hungarian(
                        sample, cfg, model, device,
                        seeds=TTA_SEEDS,
                        min_passes_for_keep=TTA_MIN_PASSES,
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as e:
                    import traceback
                    print(f"  TTA failed for {order_id}:\n{traceback.format_exc()}")
                    pred_v, pred_e = empty_solution()
                    pred_status = "tta_failed"
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            else:
                # Single-seed inference (legacy path, kept for easy revert).
                fused = fuse_and_sample(sample, cfg, rng)
                n_fused_pts = len(fused["xyz_norm"]) if fused is not None else 0
                if fused is None:
                    pred_v, pred_e = empty_solution()
                    pred_status = "fuse_failed"
                else:
                    try:
                        pred_v, pred_e = predict_sample(fused, model, device)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception as e:
                        import traceback
                        print(f"  Predict failed for {order_id}:\n{traceback.format_exc()}")
                        pred_v, pred_e = empty_solution()
                        pred_status = "predict_failed"
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

            if pred_status == "ok":
                try:
                    # Apply handcrafted triangulation tracking to catch missing corners/edges
                    try:
                        from triangulation import predict_wireframe_tracks
                        # Use min_views=3 for highly precise, conservative geometric tracks
                        track_v, track_e = predict_wireframe_tracks(sample, min_views=3)
                        track_v_count = len(track_v) if track_v is not None else 0
                        track_e_count = len(track_e) if track_e is not None else 0

                        pred_v, pred_e = hybrid_merge(pred_v, pred_e, track_v, track_e, merge_radius=0.8)
                    except Exception as track_e_err:
                        print(f"  Track ensemble failed for {order_id}: {track_e_err}")
                        pred_status = "track_failed"

                    # Vertex view-projection refinement. For each predicted 3D
                    # vertex, find the nearest gestalt-corner pixel in each
                    # view, re-triangulate via DLT, and replace the vertex if
                    # the refined position is close (<=0.5m) and reprojects
                    # well (<=10px). Local 100-sample A/B: +0.007 hss_mean,
                    # 56 wins / 41 losses vs baseline.
                    try:
                        from vertex_refine import refine_vertices_view_projection
                        pred_v, pred_e = refine_vertices_view_projection(
                            pred_v, pred_e, sample,
                            max_pixel_dist=15.0, min_views=2,
                            max_move_meters=0.5, max_reproj_px=10.0,
                        )
                    except Exception as ref_err:
                        print(f"  vertex refine failed for {order_id}: {ref_err}")

                    # Drop orphan vertices (vertices with no incident edges).
                    # Local 100-sample A/B: combined refine + orphan = +0.009
                    # hss_mean over baseline (t=1.69).
                    edges_before_2d = len(pred_e) if hasattr(pred_e, '__len__') else 0
                    try:
                        from edge_2d_filter import drop_orphan_vertices
                        pred_v, pred_e = drop_orphan_vertices(pred_v, pred_e)
                    except Exception as filt_err:
                        print(f"  orphan drop failed for {order_id}: {filt_err}")
                    edges_after_2d = len(pred_e) if hasattr(pred_e, '__len__') else 0

                    # Vertex position regressor (DINOv2): predict per-vertex 3D
                    # offset toward nearest learned position. Local 200-sample
                    # A/B: +0.0065 hss_mean on top of edge classifier (t=+2.39).
                    if vertex_regressor_bundle is not None:
                        try:
                            from vertex_regressor_v4 import refine_vertices_with_regressor
                            vr = vertex_regressor_bundle
                            pred_v, pred_e = refine_vertices_with_regressor(
                                pred_v, pred_e, sample,
                                vr["model"], vr["dino"], device=vr["dino_device"],
                                feature_mean=vr["mean"], feature_std=vr["std"],
                                edge_feat_mean=vr["edge_feat_mean"],
                                edge_feat_std=vr["edge_feat_std"],
                                max_move_meters=vr["max_move_meters"],
                            )
                        except Exception as vr_err:
                            print(f"  vertex regressor failed for {order_id}: {vr_err}")

                    # Edge classifier v4 (DINOv2 + geom features): drop the bottom
                    # ~15% of edges whose learned P(keep) is lowest. 200-sample
                    # local A/B: +0.0030 hss_mean (t=+1.26).
                    if edge_classifier_bundle is not None:
                        try:
                            from edge_classifier_v4 import classify_edges_v4
                            from edge_2d_filter import drop_orphan_vertices as _drop_orph
                            ec = edge_classifier_bundle
                            pred_v, pred_e = classify_edges_v4(
                                pred_v, pred_e, sample,
                                ec["model"], ec["dino"], device=ec["dino_device"],
                                threshold=ec["threshold"],
                                feature_mean=ec["mean"], feature_std=ec["std"],
                                edge_feat_mean=ec["edge_feat_mean"],
                                edge_feat_std=ec["edge_feat_std"],
                                min_keep_frac=ec["min_keep_frac"],
                            )
                            # Re-run orphan drop in case the classifier left orphans
                            pred_v, pred_e = _drop_orph(pred_v, pred_e)
                        except Exception as ec_err:
                            print(f"  edge classifier v4 failed for {order_id}: {ec_err}")

                except Exception as e:
                    import traceback
                    print(f"  Predict failed for {order_id}:\n{traceback.format_exc()}")
                    pred_v, pred_e = empty_solution()
                    pred_status = "predict_failed"
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            n_pred_v = len(pred_v) if hasattr(pred_v, '__len__') else 0
            n_pred_e = len(pred_e) if hasattr(pred_e, '__len__') else 0
            edges_before = locals().get('edges_before_2d', n_pred_e)
            edges_after = locals().get('edges_after_2d', n_pred_e)
            print(
                f"[DIAG] order_id={order_id} colmap={n_colmap_pts} fused={n_fused_pts} "
                f"track_v={track_v_count} track_e={track_e_count} "
                f"pred_v={n_pred_v} pred_e={n_pred_e} "
                f"2dfilt={edges_before}->{edges_after} status={pred_status}"
            )

            solution.append({
                "order_id": order_id,
                "wf_vertices": pred_v.tolist() if isinstance(pred_v, np.ndarray) else pred_v,
                "wf_edges": [(int(a), int(b)) for a, b in pred_e],
            })
            processed += 1

            if processed % 50 == 0:
                elapsed = time.time() - t_start
                rate = elapsed / processed
                remaining = (total_samples - processed) * rate
                print(f"  [{processed}/{total_samples}] "
                      f"{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining")

    # Save
    output_path = Path(params.get('output_path', '.'))
    with open(output_path / "submission.json", "w") as f:
        json.dump(solution, f)
        
    try:
        import pandas as pd
        sub = pd.DataFrame(solution, columns=["order_id", "wf_vertices", "wf_edges"])
        sub.to_parquet(output_path / "submission.parquet")
    except Exception as e:
        print(f"Failed to write parquet: {e}")

    elapsed = time.time() - t_start
    print(f"\nDone. {processed} samples in {elapsed:.0f}s ({elapsed/max(processed,1):.1f}s/sample)")
    print(f"Saved submission.json ({len(solution)} entries)")
