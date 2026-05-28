# S23DR-Ivanyshyn

My submission to the [S23DR 2026](https://huggingface.co/usm3d/) challenge
(building wireframe reconstruction from multi-view images + sparse COLMAP
point clouds). Final leaderboard score: **0.4844 HSS** (corner_f1 = 0.5298,
edge_iou = 0.4583).

## Method overview

The pipeline takes a learned 3D-segment predictor (a Perceiver transformer
over fused point clouds) and adds two **frozen-DINOv2-based learned
post-processing heads** that refine its output:

1. **Vertex position regressor** — for each predicted 3D vertex, project it
   into every COLMAP view, bilinearly sample DINOv2-small patch features at
   the projected pixel, mean+max-pool across views (→ 768-dim), concat with
   per-vertex geometric features (10-dim), and feed through a 4-layer MLP
   that predicts a 3D position offset. Offset is clamped to 0.3 m and only
   applied to vertices visible in ≥ 2 views.

2. **Edge keep/drop classifier** — for each predicted edge, sample DINOv2
   features at the projected edge midpoint per view, pool (→ 768-dim),
   concat with 40-dim geometric / multi-view-mask features, and feed
   through an MLP that predicts P(edge is a true GT edge). Drops the
   bottom ~15% of edges with the lowest classifier scores, then re-runs
   orphan-vertex cleanup.

Both heads use the **same frozen DINOv2-ViT-S/14 backbone** (22 M params,
pretrained, no fine-tuning). The trainable parts are tiny:

| Head | Params |
|---|---|
| Edge classifier | 128 K |
| Vertex regressor | 60 K |
| DINOv2 (frozen) | 22.1 M |

## Pipeline

```
fuse views → 4096 sampled points → Perceiver model → segment predictions
   → confidence filter (CONF_THRESH = 0.4)
   → segments → vertices/edges, then merge close vertices
   → snap vertices to point-cloud corner classes (apex / eave / flashing)
   → snap near-horizontal edges to horizontal
   → hybrid_merge with multi-view triangulation tracks (min_views = 3)
   → heuristic vertex refinement (snap to nearest gestalt corner pixel,
     DLT re-triangulate when ≥ 2 views agree)
   → drop orphan vertices
   → vertex position regressor (DINOv2)         ← new
   → edge keep/drop classifier (DINOv2)         ← new
   → drop orphan vertices again
```

## Score progression on held-out leaderboard

| Configuration | hss_mean |
|---|---|
| Baseline Perceiver pipeline (proven 0.4584 before view-projection refinement) | 0.4584 |
| + heuristic vertex refinement + orphan drop | 0.4815 |
| + DINOv2 edge classifier | 0.4829 |
| + DINOv2 vertex regressor | **0.4844** |

## Repository layout

```
script.py                       — inference entry point
app.py                          — HF Space stub
params.json                     — eval config
edge_classifier_v4_400.pt       — trained edge classifier head (525 KB)
vertex_regressor_v4_800.pt      — trained vertex regressor head (510 KB)
edge_classifier_v2.py           — handcrafted feature extractor (40-dim)
edge_classifier_v4.py           — edge classifier model + DINOv2 wiring
vertex_classifier_v4.py         — per-vertex feature extractor (shared)
vertex_regressor_v4.py          — vertex regressor model + inference
vertex_refine.py                — heuristic gestalt-corner snap
edge_2d_filter.py               — orphan-vertex cleanup
triangulation.py                — multi-view track builder (hybrid_merge)
mvs_utils.py                    — projection / DLT triangulation
s23dr_2026_example/             — Perceiver model architecture and utils
```

## Setup

```bash
pip install torch torchvision opencv-python scipy scikit-learn datasets \
            pillow xformers pycolmap
```

The base Perceiver checkpoint (`checkpoint.pt`, ~100 MB) is not committed.
Fetch it once before running inference:

```bash
wget https://huggingface.co/xsponenta/s23-model/resolve/main/checkpoint.pt
```

DINOv2-small weights are downloaded automatically at first run via
`torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')`.

## Training the heads

Both heads were trained on samples from `usm3d/hoho22k_2026_trainval`.
For each scene, the Perceiver model produces candidate vertices and edges,
DINOv2 features are extracted at the relevant 2D positions, and labels are
generated against the ground-truth wireframe (within a 0.4–0.5 m match
radius). Heads train in a few minutes on CPU — the heavy work is the
DINOv2 forward passes during feature collection.

| Head | Training samples | Val acc / loss |
|---|---|---|
| Edge classifier | 400 (7402 edges, 34% positive) | 82.1 % val acc |
| Vertex regressor | 800 (16 398 vertices) | best val MSE 0.032 |

## License

Apache 2.0, inherited from the upstream
[`usm3d/s23dr-2026-submission`](https://huggingface.co/jacklangerman/s23dr-2026-submission)
baseline. See `LICENSE.md`.
