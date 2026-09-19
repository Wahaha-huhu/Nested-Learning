"""Measure throughput / peak memory of an E06 arm on random tokens, BEFORE committing to a run.

Prints tokens/s, seconds per optimiser step, and the projected hours and USD for the full
token budget (docs/COST_AND_FEASIBILITY.md: "profile first, then use measured throughput").

    python -m experiments.e06_lm_from_scratch.profile_throughput --config .../configs/hope_110m.yaml
"""
import argparse
import json
import time

import torch
import torch.nn.functional as F

from nlrepro.models import build_model, count_params
from nlrepro.utils.common import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=12, help="micro-steps timed after warm-up")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    t, g = cfg["train"], cfg.get("guard", {})
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    model = build_model(cfg["model"]).to(dev)
    model.grad_checkpointing = t.get("grad_checkpointing", False)
    fwd = torch.compile(model) if t.get("compile") else model
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    V, B, L = cfg["model"]["vocab_size"], t["micro_batch"], t["seq_len"]
    torch.cuda.reset_peak_memory_stats()
    for i in range(a.warmup + a.steps):
        if i == a.warmup:
            torch.cuda.synchronize()
            t0 = time.time()
        x = torch.randint(0, V, (B, L + 1), device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = fwd(x[:, :-1])
        F.cross_entropy(logits.float().view(-1, V), x[:, 1:].reshape(-1)).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    per_micro = (time.time() - t0) / a.steps
    tok_s = B * L / per_micro
    hours = t["train_tokens"] / tok_s / 3600
    price = g.get("usd_per_hour", 0.0)
    rep = {"arch": cfg["model"]["arch"], "params": count_params(model),
           "gpu": torch.cuda.get_device_name(0), "tokens_per_s": round(tok_s),
           "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
           "projected_train_hours": round(hours, 2),
           "projected_train_usd": round(hours * price, 2), "usd_per_hour": price,
           "note": "random tokens, excludes eval/checkpoint time; add 20-30% contingency"}
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
