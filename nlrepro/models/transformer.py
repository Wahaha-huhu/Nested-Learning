"""Transformer++ baseline (Llama-style), the Table 2 'Transformer' row at small scale."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RMSNorm, Rotary, SwiGLU, init_weights


@dataclass
class TransformerConfig:
    vocab_size: int = 32000
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    ffn_hidden: int = 2048
    max_len: int = 4096
    tie_embeddings: bool = True
    rope_base: float = 10000.0


class Attention(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.h = cfg.n_heads
        self.dh = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.o = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o.is_residual_out = True
        self.rope = Rotary(self.dh, cfg.max_len, cfg.rope_base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k = self.rope(q), self.rope(k)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(y.transpose(1, 2).reshape(B, L, D))


class Block(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.n2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model, cfg.ffn_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))


class TransformerLM(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        init_weights(self, cfg.n_layers)
        if cfg.tie_embeddings:
            self.head.weight = self.embed.weight
        self.grad_checkpointing = False

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.embed(idx)
        for b in self.blocks:
            if self.grad_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(b, x, use_reentrant=False)
            else:
                x = b(x)
        return self.head(self.norm(x))
