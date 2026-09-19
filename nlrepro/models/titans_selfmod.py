"""Self-modifying Titans (paper Sec. 8.1-8.2), matrix-valued reconstruction.

What the paper specifies (Sec. 8.1-8.3, Eq. 92):
  * memories M_box for box in {k, v, q, eta, alpha, memory}, each updated in context with
        M_t = M_{t-1} (alpha_t I - eta_t k_t k_t^T) - eta_t v_hat_{box,t} k_t^T,
    where v_hat_{box,t} = M_{box,t-1}(v_t) are *self-generated* values;
  * chunk-wise training with two chunk sizes (one for M_memory, one for the others);
  * L2-normalised keys/queries and a width-4 local convolution.

What we reconstruct / simplify (all listed in docs/DEVIATIONS.md, D-HOPE-*):
  * D-HOPE-1  Matrix-valued memories (the paper's "simplest case", Eq. 92) instead of 2-layer
              residual MLP memories. This admits an exact intra-chunk dual form.
  * D-HOPE-2  Sign: we use the L2/delta-rule sign, M <- alpha-decay - eta (M k - v_hat) k^T.
              The printed "- eta v_hat k^T" would make the memory anti-learn its targets.
  * D-HOPE-3  Retention decays toward the meta-learned initial state M0 (M = M0 + D, D decays),
              not toward 0. With self-generated values, decay-to-0 collapses the projections.
  * D-HOPE-4  q, eta, alpha are produced by static (outer-level) projections; only k, v and the
              memory are self-modifying. eta and alpha are data-dependent per head and token.
  * D-HOPE-5  M_memory uses the standard value v (Titans); M_k, M_v use self-generated values.
              `self_values_memory=True` switches the memory to self-generated values as well.
  * D-HOPE-6  Mini-batch semantics inside a chunk: all errors are evaluated at the chunk-start
              state (as in Titans/TTT chunked training); the memory read is exact w.r.t. those
              updates (dual form), the projections k, v use the chunk-start state.
  * D-HOPE-7  One chunk size for all memories by default (proj_every multiplies it for M_k, M_v).

Row-vector convention: a memory M (d_out x d_in) maps x -> M x; batched rows use X @ M^T.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import CausalDepthwiseConv, RMSNorm, l2norm


@dataclass
class TitansConfig:
    d_model: int = 768
    n_heads: int = 12
    chunk: int = 64
    conv_width: int = 4
    eta_max: float = 0.2          # per-token inner step size is eta_max * sigmoid(.)
    alpha_bias: float = 5.0       # retention init: sigmoid(5) ~ 0.9933 per token
    proj_lr_scale: float = 0.1    # inner step-size multiplier for M_k, M_v
    proj_every: int = 1           # M_k, M_v update every proj_every memory chunks
    self_modifying: bool = True   # False: M_k, M_v stay at their meta-learned init (static)
    self_values_memory: bool = False


def _chunk_terms(Lam: torch.Tensor):
    """Lam: (B,H,c) inclusive cumulative log-retention inside the chunk."""
    c = Lam.shape[-1]
    diff = Lam.unsqueeze(-1) - Lam.unsqueeze(-2)                     # (B,H,t,s)
    mask = torch.ones(c, c, dtype=torch.bool, device=Lam.device).tril()
    P = torch.exp(diff.masked_fill(~mask, float("-inf")))             # e^{Lam_t - Lam_s}, s<=t
    w_end = torch.exp(Lam[..., -1:] - Lam)                            # e^{Lam_C - Lam_s}
    decay_end = torch.exp(Lam[..., -1])[..., None, None]              # e^{Lam_C}
    return P, w_end, decay_end


def selfmod_titans_scan(u, q, eta, log_alpha, M0, chunk, proj_lr_scale=0.1, proj_every=1,
                        self_modifying=True, self_values_memory=False):
    """Chunk-parallel self-modifying Titans.

    u, q: (B,H,L,d) inputs to the projection memories and (normalised) queries
    eta, log_alpha: (B,H,L) per-token inner step size and log retention
    M0: (3,H,d,d) meta-learned initial states for (M_k, M_v, M_memory)
    returns Y: (B,H,L,d)
    """
    B, H, L, d = u.shape
    M0k, M0v, M0m = (M0[i].unsqueeze(0) for i in range(3))
    Dk = u.new_zeros(B, H, d, d)
    Dv = u.new_zeros(B, H, d, d)
    Dm = u.new_zeros(B, H, d, d)
    acc_k = acc_v = None
    ys = []
    for n, c0 in enumerate(range(0, L, chunk)):
        sl = slice(c0, min(c0 + chunk, L))
        U, Q, et = u[:, :, sl], q[:, :, sl], eta[:, :, sl]
        Lam = torch.cumsum(log_alpha[:, :, sl], dim=-1)
        P, w_end, decay_end = _chunk_terms(Lam)

        Mk, Mv, Mm = M0k + Dk, M0v + Dv, M0m + Dm
        K = l2norm(U @ Mk.transpose(-1, -2))
        V = U @ Mv.transpose(-1, -2)
        Vhat = V @ Mm.transpose(-1, -2) if self_values_memory else V
        E = K @ Mm.transpose(-1, -2) - Vhat                           # errors at chunk start
        A = (Q @ K.transpose(-1, -2)) * P
        Y = (Q @ M0m.transpose(-1, -2)
             + torch.exp(Lam).unsqueeze(-1) * (Q @ Dm.transpose(-1, -2))
             - A @ (et.unsqueeze(-1) * E))
        ys.append(Y)

        w = (w_end * et).unsqueeze(-1)
        Dm = decay_end * Dm - (w * E).transpose(-1, -2) @ K
        if self_modifying:
            # self-generated values v_hat_box = M_box v  ->  error M_box (k - v)
            KV = K - V
            gk = (w * (KV @ Mk.transpose(-1, -2))).transpose(-1, -2) @ K
            gv = (w * (KV @ Mv.transpose(-1, -2))).transpose(-1, -2) @ K
            if proj_every == 1:
                Dk = decay_end * Dk - proj_lr_scale * gk
                Dv = decay_end * Dv - proj_lr_scale * gv
            else:  # accumulate and apply every proj_every chunks (lower frequency)
                acc_k = gk if acc_k is None else acc_k + gk
                acc_v = gv if acc_v is None else acc_v + gv
                Dk, Dv = decay_end * Dk, decay_end * Dv
                if (n + 1) % proj_every == 0:
                    Dk = Dk - proj_lr_scale * acc_k
                    Dv = Dv - proj_lr_scale * acc_v
                    acc_k = acc_v = None
    return torch.cat(ys, dim=2)


@torch.no_grad()
def selfmod_titans_reference(u, q, eta, log_alpha, M0, chunk, proj_lr_scale=0.1,
                             self_modifying=True, self_values_memory=False):
    """Token-by-token loop with identical (mini-batch) semantics. For tests only (proj_every=1)."""
    B, H, L, d = u.shape
    out = torch.zeros_like(u)
    for b in range(B):
        for h in range(H):
            M0k, M0v, M0m = M0[0, h], M0[1, h], M0[2, h]
            Dk = torch.zeros(d, d, dtype=u.dtype)
            Dv = torch.zeros_like(Dk)
            Dm = torch.zeros_like(Dk)
            for c0 in range(0, L, chunk):
                Mk, Mv, Mm = M0k + Dk, M0v + Dv, M0m + Dm       # frozen for the chunk
                for t in range(c0, min(c0 + chunk, L)):
                    a = torch.exp(log_alpha[b, h, t])
                    k = Mk @ u[b, h, t]
                    k = k / torch.sqrt((k * k).sum() + 1e-6)
                    v = Mv @ u[b, h, t]
                    vhat = Mm @ v if self_values_memory else v
                    e = Mm @ k - vhat
                    Dm = a * Dm - eta[b, h, t] * torch.outer(e, k)
                    out[b, h, t] = (M0m + Dm) @ q[b, h, t]
                    if self_modifying:
                        Dk = a * Dk - proj_lr_scale * eta[b, h, t] * torch.outer(Mk @ (k - v), k)
                        Dv = a * Dv - proj_lr_scale * eta[b, h, t] * torch.outer(Mv @ (k - v), k)
    return out


class SelfModTitans(nn.Module):
    """Sequence mixer used in place of attention in a Hope block."""

    def __init__(self, cfg: TitansConfig):
        super().__init__()
        self.cfg = cfg
        D, H = cfg.d_model, cfg.n_heads
        assert D % H == 0
        self.h, self.d = H, D // H
        self.w_u = nn.Linear(D, D, bias=False)
        self.conv = CausalDepthwiseConv(D, cfg.conv_width)
        self.w_q = nn.Linear(D, D, bias=False)
        self.w_eta = nn.Linear(D, H, bias=True)
        self.w_alpha = nn.Linear(D, H, bias=True)
        self.w_gate = nn.Linear(D, D, bias=False)
        self.out_norm = RMSNorm(self.d)
        self.w_o = nn.Linear(D, D, bias=False)
        self.w_o.is_residual_out = True
        eye = torch.eye(self.d).expand(3, H, self.d, self.d).clone()
        self.M0 = nn.Parameter(eye)   # meta-learned initial states (k, v, memory), identity init

    def reset_special_parameters(self) -> None:
        """Called after the global init (which overwrites Linear biases with zeros)."""
        nn.init.zeros_(self.w_eta.bias)
        nn.init.constant_(self.w_alpha.bias, self.cfg.alpha_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        H, d = self.h, self.d
        cdt = torch.float64 if x.dtype == torch.float64 else torch.float32   # recurrence dtype
        u = F.silu(self.conv(self.w_u(x))).view(B, L, H, d).transpose(1, 2)
        q = l2norm(self.w_q(x).view(B, L, H, d).transpose(1, 2).to(cdt))
        eta = self.cfg.eta_max * torch.sigmoid(self.w_eta(x).to(cdt)).transpose(1, 2)
        log_alpha = F.logsigmoid(self.w_alpha(x).to(cdt)).transpose(1, 2)
        with torch.autocast(device_type=x.device.type, enabled=False):
            y = selfmod_titans_scan(
                u.to(cdt), q, eta, log_alpha, self.M0.to(cdt), self.cfg.chunk,
                self.cfg.proj_lr_scale, self.cfg.proj_every, self.cfg.self_modifying,
                self.cfg.self_values_memory)
        y = self.out_norm(y.transpose(1, 2)).reshape(B, L, H * d).to(x.dtype)
        return self.w_o(y * F.silu(self.w_gate(x)))
