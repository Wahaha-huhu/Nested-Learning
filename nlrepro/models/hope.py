"""Hope (paper Sec. 8.3): self-modifying Titans followed by a Continuum Memory System.

Block:  x <- x + SelfModTitans(norm(x));  x <- CMS(x)  (CMS applies its own norms/residuals).
No positional encoding: the mixer is recurrent. See titans_selfmod.py and cms.py for the
reconstruction notes (docs/DEVIATIONS.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .cms import CMS, CMSConfig, CMSLevelSpec
from .layers import RMSNorm, init_weights
from .titans_selfmod import SelfModTitans, TitansConfig


@dataclass
class HopeConfig:
    vocab_size: int = 32000
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    tie_embeddings: bool = True
    titans: dict = field(default_factory=dict)       # overrides for TitansConfig
    cms_levels: list = field(default_factory=lambda: [
        {"kind": "incontext", "chunk": 512, "hidden_mult": 2, "lr": 1.0},
        {"kind": "static", "hidden": 1152},
    ])


class HopeBlock(nn.Module):
    def __init__(self, cfg: HopeConfig):
        super().__init__()
        tcfg = TitansConfig(d_model=cfg.d_model, n_heads=cfg.n_heads, **cfg.titans)
        self.n1 = RMSNorm(cfg.d_model)
        self.mixer = SelfModTitans(tcfg)
        self.cms = CMS(cfg.d_model, cfg.n_heads,
                       CMSConfig(levels=[CMSLevelSpec(**s) for s in cfg.cms_levels]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.n1(x))
        return self.cms(x)


class HopeLM(nn.Module):
    def __init__(self, cfg: HopeConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([HopeBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        init_weights(self, cfg.n_layers)
        for m in self.modules():
            if hasattr(m, "reset_special_parameters"):
                m.reset_special_parameters()
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
