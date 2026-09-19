"""Model factory. `build_model({"arch": "transformer"|"hope", ...})`."""
from __future__ import annotations

from .hope import HopeConfig, HopeLM
from .transformer import TransformerConfig, TransformerLM


def build_model(cfg: dict):
    cfg = dict(cfg)
    arch = cfg.pop("arch")
    if arch == "transformer":
        return TransformerLM(TransformerConfig(**cfg))
    if arch == "hope":
        return HopeLM(HopeConfig(**cfg))
    raise ValueError(f"unknown arch {arch}")


def count_params(model) -> dict:
    """Unique parameters (tied weights counted once)."""
    seen, total, emb = set(), 0, 0
    for name, p in model.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
        if "embed" in name:
            emb += p.numel()
    return {"total": total, "embedding": emb, "non_embedding": total - emb}
