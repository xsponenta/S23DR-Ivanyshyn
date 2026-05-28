import torch

from .wire_varifold_kernels import (
    loss_simpson3_batch,
    loss_simpson3_mix_batch,
)


def segments_to_vertices_edges(segments: torch.Tensor):
    segs = torch.as_tensor(segments, dtype=torch.float32)
    vertices = segs.reshape(-1, 3)
    edges = [(2 * i, 2 * i + 1) for i in range(segs.shape[0])]
    return vertices, edges


def varifold_loss_batch(
    pred_segments: torch.Tensor,
    gt_segments: torch.Tensor,
    *,
    sigma: float = 0.1,
    variant: str = "semi_lobatto3",
    t_nodes01: torch.Tensor | None = None,
    t_w: torch.Tensor | None = None,
    sigmas: torch.Tensor | None = None,
    alpha: torch.Tensor | None = None,
    normalize_alpha: bool = True,
    len_pow: float | None = None,
    gt_mask: torch.Tensor | None = None,
    pred_weights: torch.Tensor | None = None,
    cross_only: bool = False,
) -> torch.Tensor:
    if pred_segments.dim() != 4 or gt_segments.dim() != 4:
        raise ValueError("pred_segments and gt_segments must be (B, N, 2, 3)")
    p_pred, q_pred = pred_segments[:, :, 0], pred_segments[:, :, 1]
    p_gt, q_gt = gt_segments[:, :, 0], gt_segments[:, :, 1]

    w_gt = None
    if gt_mask is not None:
        w_gt = gt_mask.to(device=pred_segments.device, dtype=pred_segments.dtype)

    w_pred = None
    if pred_weights is not None:
        w_pred = pred_weights.to(device=pred_segments.device, dtype=pred_segments.dtype)

    if variant != "simpson3":
        raise ValueError(
            f"Unsupported varifold variant: {variant!r}. "
            f"Only 'simpson3' is supported in batch mode.")
    if sigmas is not None or alpha is not None:
        if sigmas is None or alpha is None:
            raise ValueError("sigmas and alpha are required for simpson3 mix")
        return loss_simpson3_mix_batch(p_pred, q_pred, p_gt, q_gt, sigmas, alpha, w_gt=w_gt, w_pred=w_pred, normalize_alpha=normalize_alpha, cross_only=cross_only)
    return loss_simpson3_batch(p_pred, q_pred, p_gt, q_gt, sigma, w_gt=w_gt, w_pred=w_pred)
