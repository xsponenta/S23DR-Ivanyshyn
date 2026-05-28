# custom_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# Core Efficient Multihead Attention using Scaled Dot Product Attention (SDPA)
# =============================================================================

class MultiHeadSDPA(nn.Module):
    """
    Multi-head cross-attention using torch.nn.functional.scaled_dot_product_attention
    without causal masking. Suitable for set inputs and cross-attention.

    If qk_norm=True, L2-normalizes Q and K per-head before the dot product,
    then scales by a learned per-head temperature (log_scale). This caps logit
    magnitude to [-1, +1] * exp(log_scale), preventing attention entropy
    collapse at large head_dim.
    """
    def __init__(self, d_model: int, num_heads: int, kv_heads: int = None,
                 qk_norm: bool = False, qk_norm_type: str = "l2"):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.kv_heads = kv_heads or num_heads
        assert self.num_heads % self.kv_heads == 0, "kv_heads must divide num_heads"

        self.head_dim = d_model // num_heads
        self.qk_norm = qk_norm
        self.qk_norm_type = qk_norm_type

        # Input projection layers
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.kv_heads * self.head_dim, bias=False)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.out_proj.weight)

        if qk_norm:
            import math
            if qk_norm_type == "rms":
                # Standard QK-norm (Qwen3/Gemma3 style): RMSNorm on Q and K,
                # no learned temperature. SDPA's 1/sqrt(d) scaling is sufficient
                # because RMSNorm preserves the expected logit variance.
                pass  # no extra parameters needed
            else:
                # L2 + learned temperature (nGPT/ViT-22B style):
                # L2 projects to unit sphere, needs learned scale to compensate.
                self.log_scale = nn.Parameter(
                    torch.full((num_heads,), math.log(math.sqrt(self.head_dim))))

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Project
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(key)

        B, Tq, _ = q.shape
        _, Tk, _ = k.shape

        q = q.view(B, Tq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, Tk, self.kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, Tk, self.kv_heads, self.head_dim).transpose(1, 2)

        if self.kv_heads != self.num_heads:
            repeat = self.num_heads // self.kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        if self.qk_norm:
            if self.qk_norm_type == "rms":
                # RMSNorm (Qwen3/Gemma3 style): no learned temperature needed.
                # After RMSNorm, logit variance matches standard SDPA naturally.
                q = q * torch.rsqrt(q.square().mean(dim=-1, keepdim=True) + 1e-6)
                k = k * torch.rsqrt(k.square().mean(dim=-1, keepdim=True) + 1e-6)
                attn_mask = None
                if key_padding_mask is not None:
                    attn_mask = ~key_padding_mask[:, None, None, :].to(dtype=torch.bool)
                attn_out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False,
                )
            else:
                # L2 + learned temperature (nGPT/ViT-22B style)
                q = F.normalize(q, dim=-1)
                k = F.normalize(k, dim=-1)
                scale = self.log_scale.exp().view(1, -1, 1, 1)
                q = q * scale
                attn_mask = None
                if key_padding_mask is not None:
                    attn_mask = ~key_padding_mask[:, None, None, :].to(dtype=torch.bool)
                attn_out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False,
                    scale=1.0,
                )
        else:
            attn_mask = None
            if key_padding_mask is not None:
                attn_mask = ~key_padding_mask[:, None, None, :].to(dtype=torch.bool)
            attn_out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
            )

        attn_out = attn_out.transpose(1, 2).reshape(B, Tq, self.d_model)
        return self.out_proj(attn_out)


# =============================================================================
# Transformer Feed-Forward Block
# =============================================================================

def _get_activation(name: str):
    """Look up activation function by name. Supports 'relu_sq' for ReLU^2."""
    if name == "relu_sq":
        return lambda x: F.relu(x).square()
    return getattr(F, name)


class FeedForward(nn.Module):
    """
    Position-wise MLP block: linear -> activation -> linear.
    Supports 'gelu', 'relu', 'relu_sq', etc.
    """
    def __init__(self, d_model: int, dim_ff: int, activation: str = "gelu"):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.linear2 = nn.Linear(dim_ff, d_model)
        nn.init.zeros_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)
        self.activation = _get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        return self.linear2(self.activation(x))
