"""Small shared layers (Llama-style 'Transformer++' conventions)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dt)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)
        self.down.is_residual_out = True  # scaled init marker

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Rotary(nn.Module):
    def __init__(self, head_dim: int, max_len: int = 8192, base: float = 10000.0):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_len).float()
        f = torch.outer(t, inv)
        self.register_buffer("cos", f.cos(), persistent=False)
        self.register_buffer("sin", f.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, L, D)
        L = x.shape[-2]
        if L > self.cos.shape[0]:
            raise ValueError(f"sequence length {L} exceeds rotary cache {self.cos.shape[0]}")
        cos = self.cos[:L].to(x.dtype)
        sin = self.sin[:L].to(x.dtype)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
        return out.flatten(-2)


class CausalDepthwiseConv(nn.Module):
    """Short causal depthwise convolution (width 4 in the paper, Sec. 8.3)."""

    def __init__(self, channels: int, width: int = 4):
        super().__init__()
        self.width = width
        self.conv = nn.Conv1d(channels, channels, width, groups=channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        y = F.pad(x.transpose(1, 2), (self.width - 1, 0))
        return self.conv(y).transpose(1, 2)


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + eps)


def init_weights(module: nn.Module, n_layers: int, std: float = 0.02) -> None:
    """GPT-2/Llama-style init; residual output projections scaled by 1/sqrt(2 n_layers)."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            s = std / math.sqrt(2 * n_layers) if getattr(m, "is_residual_out", False) else std
            nn.init.normal_(m.weight, mean=0.0, std=s)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=std)
