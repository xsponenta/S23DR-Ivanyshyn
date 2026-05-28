"""
Perceiver-based transformer for 3D roof wireframe prediction.

Architecture overview:

    Input tokens [B, T, D]
        |
        v
    input_proj: Linear -> GELU -> Linear -> LayerNorm   =>  [B, T, hidden]
        |
        v
    Perceiver latent bottleneck (N PerceiverLatentLayers):
        Learnable latent embeddings [L, hidden] are broadcast to batch.
        Each layer: cross-attn(latents <- tokens) -> self-attn(latents) -> FFN
        Output: latents [B, L, hidden]
        |
        v
    Segment decoder (M SegmentDecoderLayers):
        Learnable query embeddings [S, hidden] are broadcast to batch.
        Each layer: cross-attn(queries <- latents) -> self-attn(queries) -> FFN
        Output: queries [B, S, hidden]
        |
        v
    segment_head: Linear -> 6D -> (midpoint, half_vector)
        + query_offsets (learnable per-query bias)
        endpoints = midpoint +/- half_vector  ->  [B, S, 2, 3]
"""

import torch
import torch.nn as nn

from .attention import MultiHeadSDPA, FeedForward


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class AttnResidual(nn.Module):
    """Pre-norm attention + residual + dropout."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        kv_heads: int | None = None,
        norm_class=None,
        qk_norm: bool = False,
        qk_norm_type: str = "l2",
    ):
        super().__init__()
        norm_class = norm_class or nn.LayerNorm
        self.norm = norm_class(d_model)
        self.attn = MultiHeadSDPA(d_model, num_heads, kv_heads=kv_heads, qk_norm=qk_norm, qk_norm_type=qk_norm_type)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        res = x
        x = self.norm(x)
        x = self.attn(x, memory, key_padding_mask=memory_key_padding_mask)
        return res + self.drop(x)


class FFNResidual(nn.Module):
    """Pre-norm feed-forward + residual + dropout."""

    def __init__(
        self,
        d_model: int,
        dim_ff: int,
        dropout: float = 0.0,
        activation: str = "gelu",
        norm_class=None,
    ):
        super().__init__()
        norm_class = norm_class or nn.LayerNorm
        self.norm = norm_class(d_model)
        self.ffn = FeedForward(d_model, dim_ff, activation=activation)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.norm(x)
        x = self.ffn(x)
        return res + self.drop(x)


# ---------------------------------------------------------------------------
# Perceiver encoder layer
# ---------------------------------------------------------------------------

class PerceiverLatentLayer(nn.Module):
    """Single Perceiver latent layer.

    If use_cross=True:  cross-attn(latents <- points) -> self-attn -> FFN
    If use_cross=False: self-attn -> FFN  (saves compute in deep stacks)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_ff: int,
        dropout: float = 0.0,
        activation: str = "gelu",
        kv_heads_cross: int | None = None,
        kv_heads_self: int | None = None,
        use_cross: bool = True,
        norm_class=None,
        qk_norm: bool = False,
        qk_norm_type: str = "l2",
    ):
        super().__init__()
        self.use_cross = use_cross
        if use_cross:
            self.cross = AttnResidual(d_model, num_heads, dropout, kv_heads=kv_heads_cross, norm_class=norm_class, qk_norm=qk_norm, qk_norm_type=qk_norm_type)
        self.self_attn = AttnResidual(d_model, num_heads, dropout, kv_heads=kv_heads_self, norm_class=norm_class, qk_norm=qk_norm, qk_norm_type=qk_norm_type)
        self.ffn = FFNResidual(d_model, dim_ff, dropout, activation=activation, norm_class=norm_class)

    def forward(
        self,
        latents: torch.Tensor,
        points: torch.Tensor,
        points_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_cross:
            latents = self.cross(latents, points, memory_key_padding_mask=points_key_padding_mask)
        latents = self.self_attn(latents, latents)
        latents = self.ffn(latents)
        return latents


# ---------------------------------------------------------------------------
# Segment decoder layer
# ---------------------------------------------------------------------------

class SegmentDecoderLayer(nn.Module):
    """Single segment decoder layer.

    cross-attn(queries <- latents) -> [cross-attn(queries <- inputs)] -> self-attn(queries) -> FFN

    If input_xattn=True, adds a second cross-attention that attends directly
    to the projected input tokens (bypassing the latent bottleneck). This gives
    queries access to fine-grained point-level detail for vertex precision.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_ff: int,
        dropout: float = 0.0,
        activation: str = "gelu",
        kv_heads_cross: int | None = None,
        kv_heads_self: int | None = None,
        norm_class=None,
        input_xattn: bool = False,
        qk_norm: bool = False,
        qk_norm_type: str = "l2",
    ):
        super().__init__()
        self.cross = AttnResidual(d_model, num_heads, dropout, kv_heads=kv_heads_cross, norm_class=norm_class, qk_norm=qk_norm, qk_norm_type=qk_norm_type)
        self.input_xattn = input_xattn
        if input_xattn:
            self.cross_input = AttnResidual(d_model, num_heads, dropout, kv_heads=kv_heads_cross, norm_class=norm_class, qk_norm=qk_norm, qk_norm_type=qk_norm_type)
        self.self_attn = AttnResidual(d_model, num_heads, dropout, kv_heads=kv_heads_self, norm_class=norm_class, qk_norm=qk_norm, qk_norm_type=qk_norm_type)
        self.ffn = FFNResidual(d_model, dim_ff, dropout, activation=activation, norm_class=norm_class)

    def forward(
        self,
        queries: torch.Tensor,
        latents: torch.Tensor,
        src: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        queries = self.cross(queries, latents)
        if self.input_xattn and src is not None:
            queries = self.cross_input(queries, src, memory_key_padding_mask=src_key_padding_mask)
        queries = self.self_attn(queries, queries)
        queries = self.ffn(queries)
        return queries


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class TokenTransformerSegments(nn.Module):
    """Perceiver transformer that predicts 3D roof wireframe segments.

    Takes point-cloud tokens and outputs segment endpoints as [B, S, 2, 3]
    where S is the number of segments and each segment has two 3D endpoints.

    Args:
        segments:       Number of predicted segments (S).
        in_dim:         Dimensionality of input tokens.
        hidden:         Internal hidden dimension throughout the model.
        num_heads:      Number of attention heads.
        kv_heads_cross: Grouped-query heads for cross-attention (None = standard MHA).
        kv_heads_self:  Grouped-query heads for self-attention (None = standard MHA).
        dim_feedforward: FFN intermediate dimension.
        dropout:        Dropout rate applied after attention and FFN.
        latent_tokens:  Number of learnable latent embeddings (L) in the bottleneck.
        latent_layers:  Number of PerceiverLatentLayers (N).
        decoder_layers: Number of SegmentDecoderLayers (M).
    """

    def __init__(
        self,
        segments: int = 32,
        in_dim: int = 128,
        hidden: int = 128,
        num_heads: int = 4,
        kv_heads_cross: int | None = 2,
        kv_heads_self: int | None = 0,
        dim_feedforward: int = 256,
        dropout: float = 0.01,
        latent_tokens: int = 64,
        latent_layers: int = 2,
        decoder_layers: int = 2,
        cross_attn_interval: int = 1,
        norm_class=None,
        activation: str = "gelu",
        segment_conf: bool = False,
        pre_encoder_layers: int = 0,
        segment_param: str = "midpoint_halfvec",
        length_floor: float = 0.0,
        decoder_input_xattn: bool = False,
        qk_norm: bool = False,
        qk_norm_type: str = "l2",
    ):
        super().__init__()
        self.segments = segments
        self.out_vertices = segments * 2
        self.segment_param = segment_param
        self.decoder_input_xattn = decoder_input_xattn
        norm_class = norm_class or nn.LayerNorm

        # Treat 0 as "use standard MHA"
        if kv_heads_cross is not None and kv_heads_cross <= 0:
            kv_heads_cross = None
        if kv_heads_self is not None and kv_heads_self <= 0:
            kv_heads_self = None

        # -- Input projection --
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, hidden),
            norm_class(hidden),
        )

        # -- Optional pre-encoder: self-attention on full token sequence --
        if pre_encoder_layers > 0:
            self.pre_encoder = nn.ModuleList([
                SelfAttentionEncoderLayer(
                    d_model=hidden,
                    num_heads=num_heads,
                    dim_ff=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                    kv_heads=kv_heads_self,
                    norm_class=norm_class,
                    qk_norm=qk_norm, qk_norm_type=qk_norm_type,
                )
                for _ in range(pre_encoder_layers)
            ])
        else:
            self.pre_encoder = None

        # -- Perceiver latent bottleneck --
        self.latent_embed = nn.Embedding(latent_tokens, hidden)
        N = latent_layers
        self.latent_layers = nn.ModuleList([
            PerceiverLatentLayer(
                d_model=hidden,
                num_heads=num_heads,
                dim_ff=dim_feedforward,
                dropout=dropout,
                activation=activation,
                kv_heads_cross=kv_heads_cross,
                kv_heads_self=kv_heads_self,
                use_cross=(i == 0) or (i == N - 1) or (i % cross_attn_interval == 0),
                norm_class=norm_class,
                qk_norm=qk_norm, qk_norm_type=qk_norm_type,
            )
            for i in range(N)
        ])

        # -- Segment decoder --
        self.query_embed = nn.Embedding(segments, hidden)
        self.decoder_layers = nn.ModuleList([
            SegmentDecoderLayer(
                d_model=hidden,
                num_heads=num_heads,
                dim_ff=dim_feedforward,
                dropout=dropout,
                activation=activation,
                kv_heads_cross=kv_heads_cross,
                kv_heads_self=kv_heads_self,
                norm_class=norm_class,
                input_xattn=decoder_input_xattn,
                qk_norm=qk_norm, qk_norm_type=qk_norm_type,
            )
            for _ in range(decoder_layers)
        ])

        # -- Output head --
        if segment_param == "midpoint_dir_len":
            self.segment_head = nn.Linear(hidden, 7)  # mid(3) + dir(3) + len(1)
        else:
            self.segment_head = nn.Linear(hidden, 6)  # mid(3) + half(3)
        self.query_offsets = nn.Parameter(torch.zeros(segments, 2, 3))

        nn.init.trunc_normal_(self.segment_head.weight, mean=0.0, std=1e-3)
        if self.segment_head.bias is not None:
            nn.init.zeros_(self.segment_head.bias)
            if segment_param == "midpoint_dir_len":
                # softplus(0.5) * 0.1 ≈ 0.097 default length in normalized space
                self.segment_head.bias.data[6] = 0.5
        nn.init.normal_(self.query_offsets, mean=0.0, std=0.05)

        # -- Optional confidence head --
        self.segment_conf = segment_conf
        if segment_conf:
            self.conf_head = nn.Linear(hidden, 1)
            nn.init.zeros_(self.conf_head.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | list]:
        """
        Args:
            tokens: Input point-cloud tokens [B, T, in_dim].
            mask:   Boolean validity mask [B, T]. True = valid token.

        Returns:
            Dict with keys:
                "vertices": [B, S*2, 3] flattened endpoints.
                "segments": [B, S, 2, 3] segment endpoints.
                "edges":    Per-batch list of (start, end) index pairs into vertices.
                "conf":     [B, S] logits (only if segment_conf=True).
        """
        B = tokens.shape[0]

        # Project input tokens
        src = self.input_proj(tokens)  # [B, T, hidden]

        # Padding mask (True where padded) for cross-attention
        pad_mask = ~mask.bool() if mask is not None else None

        # Optional pre-encoder: self-attention on full token sequence
        if self.pre_encoder is not None:
            for layer in self.pre_encoder:
                src = layer(src, key_padding_mask=pad_mask)

        # Perceiver latent bottleneck
        latents = self.latent_embed.weight.unsqueeze(0).expand(B, -1, -1)
        for layer in self.latent_layers:
            latents = layer(latents, src, points_key_padding_mask=pad_mask)

        # Segment decoder
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        for layer in self.decoder_layers:
            queries = layer(queries, latents,
                            src=src if self.decoder_input_xattn else None,
                            src_key_padding_mask=pad_mask if self.decoder_input_xattn else None)

        # Predict segments -> endpoints
        if self.segment_param == "midpoint_dir_len":
            raw = self.segment_head(queries)  # [B, S, 7]
            mid = raw[:, :, :3] + self.query_offsets[:, 0, :].unsqueeze(0)
            direction = torch.nn.functional.normalize(raw[:, :, 3:6], dim=-1)
            length = torch.nn.functional.softplus(raw[:, :, 6:7]) * 0.1
            half = direction * length * 0.5
        else:
            raw = self.segment_head(queries).view(B, self.segments, 2, 3)
            raw = raw + self.query_offsets.unsqueeze(0)
            mid, half = raw[:, :, 0], raw[:, :, 1]
        seg_params = torch.stack([mid - half, mid + half], dim=2)

        vertices = seg_params.reshape(B, self.out_vertices, 3)
        edges = [[(2 * i, 2 * i + 1) for i in range(self.segments)] for _ in range(B)]

        out = {"vertices": vertices, "segments": seg_params, "edges": edges,
               "src": src, "pad_mask": pad_mask, "queries": queries}
        if self.segment_conf:
            out["conf"] = self.conf_head(queries).squeeze(-1)  # [B, S]
        return out


# ---------------------------------------------------------------------------
# Encoder-only layer (self-attention on full token sequence)
# ---------------------------------------------------------------------------

class SelfAttentionEncoderLayer(nn.Module):
    """Single self-attention layer: self-attn(tokens) -> FFN."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_ff: int,
        dropout: float = 0.0,
        activation: str = "gelu",
        kv_heads: int | None = None,
        norm_class=None,
        qk_norm: bool = False,
        qk_norm_type: str = "l2",
    ):
        super().__init__()
        self.self_attn = AttnResidual(d_model, num_heads, dropout, kv_heads=kv_heads, norm_class=norm_class, qk_norm=qk_norm, qk_norm_type=qk_norm_type)
        self.ffn = FFNResidual(d_model, dim_ff, dropout, activation=activation, norm_class=norm_class)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.self_attn(x, x, memory_key_padding_mask=key_padding_mask)
        x = self.ffn(x)
        return x


# ---------------------------------------------------------------------------
# End-to-end model: tokenizer embeddings + perceiver
# ---------------------------------------------------------------------------

class EdgeDepthSegmentsModel(nn.Module):
    """Tokenizer embeddings + transformer for 3D roof wireframes.

    Supports two architectures via the `arch` parameter:
    - "perceiver": Perceiver latent bottleneck (default, O(L*T) attention)
    - "transformer": Standard self-attention encoder (O(T^2) attention)

    Both share the same decoder, output head, and tokenizer.
    """

    def __init__(
        self,
        seq_cfg,
        segments: int = 32,
        hidden: int = 128,
        num_heads: int = 4,
        kv_heads_cross: int | None = 2,
        kv_heads_self: int | None = 0,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        latent_tokens: int = 64,
        latent_layers: int = 1,
        decoder_layers: int = 2,
        label_emb_dim: int = 16,
        src_emb_dim: int = 2,
        behind_emb_dim: int = 8,
        fourier_seed: int = 0,
        cross_attn_interval: int = 1,
        norm_class=None,
        activation: str = "gelu",
        segment_conf: bool = False,
        use_vote_features: bool = False,
        arch: str = "perceiver",
        encoder_layers: int = 4,
        pre_encoder_layers: int = 0,
        segment_param: str = "midpoint_halfvec",
        length_floor: float = 0.0,
        decoder_input_xattn: bool = False,
        qk_norm: bool = False,
        qk_norm_type: str = "l2",
        learnable_fourier: bool = False,
    ):
        super().__init__()
        self.seq_cfg = seq_cfg

        from .tokenizer import EdgeDepthSequenceBuilder
        self.tokenizer = EdgeDepthSequenceBuilder(
            seq_cfg,
            label_emb_dim=label_emb_dim,
            src_emb_dim=src_emb_dim,
            behind_emb_dim=behind_emb_dim,
            fourier_seed=fourier_seed,
            use_vote_features=use_vote_features,
            learnable_fourier=learnable_fourier,
        )

        if arch == "transformer":
            raise ValueError(
                "arch='transformer' is no longer supported. "
                "TransformerSegments has been removed; use arch='perceiver'.")
        else:
            self.segmenter = TokenTransformerSegments(
                segments=segments,
                in_dim=self.tokenizer.out_dim,
                hidden=hidden,
                num_heads=num_heads,
                kv_heads_cross=kv_heads_cross,
                kv_heads_self=kv_heads_self,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                latent_tokens=latent_tokens,
                latent_layers=latent_layers,
                decoder_layers=decoder_layers,
                cross_attn_interval=cross_attn_interval,
                norm_class=norm_class,
                activation=activation,
                segment_conf=segment_conf,
                pre_encoder_layers=pre_encoder_layers,
                segment_param=segment_param,
                length_floor=length_floor,
                decoder_input_xattn=decoder_input_xattn,
                qk_norm=qk_norm, qk_norm_type=qk_norm_type,
            )

    def forward_tokens(self, tokens: torch.Tensor, mask: torch.Tensor):
        """Run the segmenter on pre-built token tensors."""
        return self.segmenter(tokens, mask)
