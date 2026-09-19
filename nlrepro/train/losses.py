"""Memory-lean next-token loss.

`F.cross_entropy(logits.float(), ...)` on a (16 x 2048) x 32K batch keeps ~4 GiB fp32 copies
alive for backward (the first 4090 profile OOMed on exactly this). Here the fp32 upcast and
log-softmax are computed per row-chunk and recomputed in backward (activation checkpointing),
so only the bf16 logits are stored. Numerically identical to the fp32 loss.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _ce_sum(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.float(), y, reduction="sum")


def lm_loss(logits: torch.Tensor, y: torch.Tensor, rows_per_chunk: int = 8192) -> torch.Tensor:
    """Mean cross-entropy over all positions. logits: (B, L, V), y: (B, L)."""
    flat, tgt = logits.reshape(-1, logits.size(-1)), y.reshape(-1)
    total = flat.new_zeros((), dtype=torch.float32)
    for i in range(0, flat.size(0), rows_per_chunk):
        sl = slice(i, i + rows_per_chunk)
        if flat.requires_grad:
            total = total + checkpoint(_ce_sum, flat[sl], tgt[sl], use_reentrant=False)
        else:
            total = total + _ce_sum(flat[sl], tgt[sl])
    return total / tgt.numel()
