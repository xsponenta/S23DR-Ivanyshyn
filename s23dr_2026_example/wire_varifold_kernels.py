import torch

# -----------------------------
# Helpers
# -----------------------------
def segment_geom(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-9):
    """
    p,q: (...,3)
    returns d, a, ell, u:
      d   = q - p
      a   = ||d||^2
      ell = sqrt(a + eps^2)
      u   = d / ell
    """
    d = q - p
    a = (d * d).sum(dim=-1)
    eps_val = eps
    if p.dtype in (torch.float16, torch.bfloat16):
        eps_val = max(eps, float(torch.finfo(p.dtype).eps))
    ell = torch.sqrt(a + eps_val * eps_val)
    u = d / ell.unsqueeze(-1)
    return d, a, ell, u

def sample_points(p: torch.Tensor, q: torch.Tensor, nodes01: torch.Tensor):
    # (...,3) + (K,) -> (...,K,3)
    d = q - p
    nodes = nodes01.to(device=p.device, dtype=p.dtype)
    shape = [1] * (p.dim() - 1) + [nodes.shape[0], 1]
    nodes = nodes.view(*shape)
    return p.unsqueeze(-2) + nodes * d.unsqueeze(-2)


# Fixed Lobatto-3 / Simpson nodes+weights on [0,1]
LOBATTO3_NODES = torch.tensor([0.0, 0.5, 1.0])
# LOBATTO3_W = torch.tensor([1.0/6.0, 4.0/6.0, 1.0/6.0])
LOBATTO3_W = torch.tensor([1/3, 1/3, 1/3])
LOBATTO3_W2 = LOBATTO3_W[:, None] * LOBATTO3_W[None, :]  # (3,3)


def _prepare_mix_weights(sigmas, alpha, device, dtype, normalize_alpha: bool):
    sigmas_t = torch.as_tensor(sigmas, device=device, dtype=dtype).clamp_min(1e-6)
    alpha_t = torch.as_tensor(alpha, device=device, dtype=dtype)
    if normalize_alpha:
        alpha_t = alpha_t / alpha_t.sum().clamp_min(1e-12)
    return sigmas_t, alpha_t

# -----------------------------
# Simpson-3 on both segments (3x3 product rule)
# -----------------------------
def _prep_weight(w, n: int, b: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
    if w is None:
        return None
    w = torch.as_tensor(w, device=device, dtype=dtype)
    if w.dim() == 1:
        if w.shape[0] != n:
            raise ValueError(f"weight length {w.shape[0]} != {n}")
        w = w.unsqueeze(0).expand(b, -1)
    elif w.dim() == 2:
        if w.shape[0] != b or w.shape[1] != n:
            raise ValueError(f"weight shape {tuple(w.shape)} != ({b}, {n})")
    else:
        raise ValueError("weights must be 1D or 2D")
    return w


def cross_simpson3(
    pA,
    qA,
    pB,
    qB,
    sigma: float | torch.Tensor,
    wA: torch.Tensor | None = None,
    wB: torch.Tensor | None = None,
):
    device, dtype = pA.device, pA.dtype
    batched = pA.dim() == 3
    if not batched:
        pA = pA.unsqueeze(0)
        qA = qA.unsqueeze(0)
        pB = pB.unsqueeze(0)
        qB = qB.unsqueeze(0)
    nodes = LOBATTO3_NODES.to(device=device, dtype=dtype)
    w2 = LOBATTO3_W2.to(device=device, dtype=dtype)

    bsz, nA, _ = pA.shape
    nB = pB.shape[1]
    wA = _prep_weight(wA, nA, bsz, device, dtype)
    wB = _prep_weight(wB, nB, bsz, device, dtype)

    _, _, ellA, uA = segment_geom(pA, qA)
    _, _, ellB, uB = segment_geom(pB, qB)

    XA = sample_points(pA, qA, nodes)  # (B,N,3,3)
    YB = sample_points(pB, qB, nodes)  # (B,M,3,3)

    # angular + length factors: (N,M)
    ang = torch.matmul(uA, uB.transpose(-1, -2)).pow(2)
    lenfac = ellA[:, :, None] * ellB[:, None, :]
    if wA is not None or wB is not None:
        if wA is None:
            wA = torch.ones((bsz, nA), device=device, dtype=dtype)
        if wB is None:
            wB = torch.ones((bsz, nB), device=device, dtype=dtype)
        lenfac = lenfac * (wA[:, :, None] * wB[:, None, :])

    # spatial: build (N,M,3,3) kernel via broadcasting
    diff = XA[:, :, None, :, None, :] - YB[:, None, :, None, :, :]  # (B,N,M,3,3,3)
    r2 = (diff * diff).sum(dim=-1)                                 # (B,N,M,3,3)
    sigma_t = torch.as_tensor(sigma, device=device, dtype=dtype)
    if sigma_t.ndim == 0:
        inv2s2 = 1.0 / (2.0 * sigma_t * sigma_t)
    else:
        if sigma_t.shape[0] != bsz:
            raise ValueError(f"sigma batch {sigma_t.shape[0]} != {bsz}")
        inv2s2 = (1.0 / (2.0 * sigma_t * sigma_t)).view(bsz, 1, 1, 1, 1)
    K = torch.exp(-r2 * inv2s2)                                     # (B,N,M,3,3)

    spatial = (K * w2).sum(dim=-1).sum(dim=-1)                     # (B,N,M)
    out = (ang * lenfac * spatial).sum(dim=-1).sum(dim=-1)         # (B,)
    return out[0] if not batched else out


# -----------------------------
# Batch losses
# -----------------------------

def loss_simpson3_batch(
    p_pred: torch.Tensor,
    q_pred: torch.Tensor,
    p_gt: torch.Tensor,
    q_gt: torch.Tensor,
    sigma: float | torch.Tensor,
    w_gt: torch.Tensor | None = None,
    w_pred: torch.Tensor | None = None,
    cross_only: bool = False,
) -> torch.Tensor:
    cross = cross_simpson3(p_pred, q_pred, p_gt, q_gt, sigma, wA=w_pred, wB=w_gt)
    if cross_only:
        # No self-energy: avoids O(S^2) blowup, sinkhorn handles repulsion
        return -2.0 * cross
    s_pred = cross_simpson3(p_pred, q_pred, p_pred, q_pred, sigma, wA=w_pred, wB=w_pred)
    return s_pred - 2.0 * cross


def loss_simpson3_mix_batch(
    p_pred: torch.Tensor,
    q_pred: torch.Tensor,
    p_gt: torch.Tensor,
    q_gt: torch.Tensor,
    sigmas,
    alpha,
    w_gt: torch.Tensor | None = None,
    w_pred: torch.Tensor | None = None,
    normalize_alpha: bool = True,
    cross_only: bool = False,
) -> torch.Tensor:
    device, dtype = p_pred.device, p_pred.dtype
    sigmas_t = torch.as_tensor(sigmas, device=device, dtype=dtype).clamp_min(1e-6)
    alpha_t = torch.as_tensor(alpha, device=device, dtype=dtype)
    if normalize_alpha:
        alpha_t = alpha_t / alpha_t.sum().clamp_min(1e-12)
    if sigmas_t.ndim == 1:
        losses = [loss_simpson3_batch(p_pred, q_pred, p_gt, q_gt, s, w_gt=w_gt, w_pred=w_pred, cross_only=cross_only) for s in sigmas_t]
        return (torch.stack(losses, dim=0) * alpha_t[:, None]).sum(dim=0)
    if sigmas_t.ndim == 2:
        losses = [loss_simpson3_batch(p_pred, q_pred, p_gt, q_gt, sigmas_t[:, i], w_gt=w_gt, w_pred=w_pred, cross_only=cross_only) for i in range(sigmas_t.shape[1])]
        return (torch.stack(losses, dim=0) * alpha_t[:, None]).sum(dim=0)
    raise ValueError("sigmas must be 1D or 2D for batch loss")
