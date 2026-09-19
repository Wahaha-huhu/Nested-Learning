"""Continuum Memory System (paper Sec. 7.1, Eq. 70-71) for the from-scratch Hope model.

Paper: a chain y = MLP^(f_k)(... MLP^(f_1)(x)) where the parameters of level l are updated once
every C^(l) steps with the sum of the per-step error terms of that chunk (Eq. 71); the
conventional Transformer MLP is the special case k = 1 with frequency 0.

Our levels:
  * InContextMLPLevel (chunk C): a multi-head 2-layer residual MLP memory whose weights are
    reset to meta-learned initial values at the start of every sequence and updated in context
    once per chunk (tokens inside a chunk read the pre-update weights, exactly Eq. 71's
    semantics). The paper's error term for LM is the NTP gradient; that is unavailable inside the
    forward pass, so we use a local associative L2 objective ||M(k) - v||^2 (D-CMS-1).
  * StaticMLPLevel: a SwiGLU MLP updated only by the outer optimiser (frequency 0 w.r.t. context).
Each level has its own pre-norm and residual connection (the paper's Eq. 70 omits norms).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RMSNorm, SwiGLU, l2norm


def _dsilu(z: torch.Tensor) -> torch.Tensor:
    s = torch.sigmoid(z)
    return s * (1 + z * (1 - s))


def incontext_mlp_scan(a, v, eta, log_alpha, W1_0, W2_0, chunk):
    """a (keys = read inputs), v: (B,H,L,d); eta, log_alpha: (B,H,L);
    W1_0: (H,h,d), W2_0: (H,d,h). Memory M(x) = x + W2 silu(W1 x). Returns (B,H,L,d)."""
    B, H, L, d = a.shape
    D1 = a.new_zeros(B, H, *W1_0.shape[1:])
    D2 = a.new_zeros(B, H, *W2_0.shape[1:])
    outs = []
    for c0 in range(0, L, chunk):
        sl = slice(c0, min(c0 + chunk, L))
        A, Vc, et = a[:, :, sl], v[:, :, sl], eta[:, :, sl]
        W1 = W1_0.unsqueeze(0) + D1
        W2 = W2_0.unsqueeze(0) + D2
        z = A @ W1.transpose(-1, -2)                 # (B,H,c,h)
        g = F.silu(z)
        m = A + g @ W2.transpose(-1, -2)             # read with chunk-start weights (Eq. 71)
        outs.append(m)
        e = (m - Vc) * et.unsqueeze(-1)              # eta-weighted d/dm of 0.5||m - v||^2
        gW2 = e.transpose(-1, -2) @ g                # (B,H,d,h)
        dz = (e @ W2) * _dsilu(z)                    # (B,H,c,h)
        gW1 = dz.transpose(-1, -2) @ A               # (B,H,h,d)
        ret = torch.exp(log_alpha[:, :, sl].sum(-1))[..., None, None]
        D1 = ret * D1 - gW1                          # retention toward the meta-learned init
        D2 = ret * D2 - gW2
    return torch.cat(outs, dim=2)


class InContextMLPLevel(nn.Module):
    def __init__(self, d_model: int, n_heads: int, chunk: int, hidden_mult: int = 2,
                 lr: float = 1.0, alpha_bias: float = 5.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.d, self.chunk = n_heads, d_model // n_heads, chunk
        self.eta_max = lr / chunk        # summed over a chunk ~ lr * mean gradient
        self.alpha_bias = alpha_bias
        hid = hidden_mult * self.d
        self.w_in = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_eta = nn.Linear(d_model, n_heads, bias=True)
        self.w_alpha = nn.Linear(d_model, n_heads, bias=True)
        self.w_out = nn.Linear(d_model, d_model, bias=False)
        self.w_out.is_residual_out = True
        self.W1_0 = nn.Parameter(torch.randn(n_heads, hid, self.d) / self.d ** 0.5)
        self.W2_0 = nn.Parameter(0.02 * torch.randn(n_heads, self.d, hid))

    def reset_special_parameters(self) -> None:
        nn.init.zeros_(self.w_eta.bias)
        nn.init.constant_(self.w_alpha.bias, self.alpha_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        cdt = torch.float64 if x.dtype == torch.float64 else torch.float32
        a = l2norm(self.w_in(x).view(B, L, self.h, self.d).transpose(1, 2).to(cdt))
        v = self.w_v(x).view(B, L, self.h, self.d).transpose(1, 2).to(cdt)
        eta = self.eta_max * torch.sigmoid(self.w_eta(x).to(cdt)).transpose(1, 2)
        la = F.logsigmoid(self.w_alpha(x).to(cdt)).transpose(1, 2)
        with torch.autocast(device_type=x.device.type, enabled=False):
            y = incontext_mlp_scan(a, v, eta, la, self.W1_0.to(cdt), self.W2_0.to(cdt),
                                   self.chunk)
        return self.w_out(y.transpose(1, 2).reshape(B, L, D).to(x.dtype))


@dataclass
class CMSLevelSpec:
    kind: str = "static"            # "static" | "incontext"
    hidden: int = 2048              # static: SwiGLU hidden size
    chunk: int = 512                # incontext: update period in tokens (C^(l))
    hidden_mult: int = 2            # incontext: per-head memory hidden = mult * head_dim
    lr: float = 1.0                 # incontext: inner step size on the chunk-mean gradient


@dataclass
class CMSConfig:
    levels: list = field(default_factory=lambda: [CMSLevelSpec("incontext", chunk=512),
                                                  CMSLevelSpec("static", hidden=1152)])


class CMS(nn.Module):
    """Chain of levels, highest frequency first (paper Eq. 70)."""

    def __init__(self, d_model: int, n_heads: int, cfg: CMSConfig):
        super().__init__()
        self.norms = nn.ModuleList()
        self.levels = nn.ModuleList()
        for spec in cfg.levels:
            spec = spec if isinstance(spec, CMSLevelSpec) else CMSLevelSpec(**spec)
            self.norms.append(RMSNorm(d_model))
            if spec.kind == "static":
                self.levels.append(SwiGLU(d_model, spec.hidden))
            elif spec.kind == "incontext":
                self.levels.append(InContextMLPLevel(d_model, n_heads, spec.chunk,
                                                     spec.hidden_mult, spec.lr))
            else:
                raise ValueError(spec.kind)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for n, lvl in zip(self.norms, self.levels):
            x = x + lvl(n(x))
        return x
