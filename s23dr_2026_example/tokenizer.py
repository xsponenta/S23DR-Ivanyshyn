"""Tokenizer: learned embeddings + Fourier features for the point cloud tokens.

The EdgeDepthSequenceBuilder holds the learned embedding tables (label, source,
behind) and the random Fourier positional encoding. At training time,
build_tokens() in data.py applies these to pre-sampled point indices on GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from .point_fusion import NUM_ADE, NUM_GEST


# -- Config --

@dataclass(frozen=True)
class EdgeDepthSequenceConfig:
    seq_len: int = 2048
    colmap_points: int = 1280
    depth_points: int = 768
    use_fourier: bool = True
    fourier_dim: int = 32
    fourier_scale: float = 10.0


# -- Fourier positional encoding --

class FourierFeatures(nn.Module):
    def __init__(self, in_dim: int = 3, fourier_dim: int = 64,
                 scale: float = 10.0, seed: int = 0,
                 learnable: bool = False):
        super().__init__()
        gen = torch.Generator()
        gen.manual_seed(seed)
        B = torch.randn(fourier_dim, in_dim, generator=gen) * scale
        if learnable:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer("B", B, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = (2.0 * np.pi) * (x @ self.B.t())
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


# -- Sequence builder (holds embeddings) --

class EdgeDepthSequenceBuilder(nn.Module):
    """Holds learned embeddings for point cloud tokenization.

    Used by the model at training time: build_tokens() calls
    self.label_emb(class_id), self.src_emb(source), etc.
    """

    def __init__(self, cfg: EdgeDepthSequenceConfig, label_emb_dim: int = 16,
                 src_emb_dim: int = 2, behind_emb_dim: int = 8,
                 fourier_seed: int = 0, use_vote_features: bool = False,
                 learnable_fourier: bool = False):
        super().__init__()
        self.cfg = cfg

        self.num_labels = 13  # 11 structural + other_house + non_house
        self.label_emb = nn.Embedding(self.num_labels, label_emb_dim)
        self.src_emb = nn.Embedding(2, src_emb_dim)
        self.behind_emb_dim = behind_emb_dim
        if behind_emb_dim > 0:
            self.behind_emb = nn.Embedding(NUM_GEST + 1, behind_emb_dim)

        # Fourier positional encoding
        if cfg.use_fourier:
            self.pos_enc = FourierFeatures(
                in_dim=3, fourier_dim=cfg.fourier_dim,
                scale=cfg.fourier_scale, seed=fourier_seed,
                learnable=learnable_fourier,
            )
            pos_dim = 3 + 2 * cfg.fourier_dim
        else:
            self.pos_enc = None
            pos_dim = 3

        vote_dim = 2 if use_vote_features else 0  # n_views_voted + vote_frac
        self.use_vote_features = use_vote_features
        self.out_dim = pos_dim + label_emb_dim + src_emb_dim + behind_emb_dim + vote_dim
