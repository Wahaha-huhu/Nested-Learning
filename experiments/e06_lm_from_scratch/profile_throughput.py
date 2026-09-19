"""Measure throughput / peak memory of an E06 arm on random tokens, BEFORE committing to a run.

Searches the largest micro-batch that fits (configured value, then halving; activation
checkpointing as a last resort), keeping tokens per optimiser step fixed via grad_accum, so the
optimisation is identical for every micro-batch. Prints the projected hours/USD for the full
token budget and the exact overrides to pass to train.py.

    python -m experiments.e06_lm_from_scratch.profile_throughput --config .../configs/hope_110m.yaml
"""
import argparse
import gc
import json
import time

import torch

from nlrepro.models import build_model, count_params
from nlrepro.train.losses import lm_loss
from nlrepro.utils.common import load_config


def _try(cfg, micro, ckpt, steps, warmup):
    t = cfg["train"]
    model = build_model(cfg["model"]).cuda()
    model.grad_checkpointing = ckpt
    fwd = torch.compile(model) if t.get("compile") else model
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    V, L = cfg["model"]["vocab_size"], t["seq_len"]
    params = count_params(model)
    torch.cuda.reset_peak_memory_stats()
    out = {"ok": False}
    try:
        for i in range(warmup + steps):
            if i == warmup:
                torch.cuda.synchronize()
                t0 = time.time()
            x = torch.randint(0, V, (micro, L + 1), device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = fwd(x[:, :-1])
            lm_loss(logits, x[:, 1:]).backward()
            del logits
            opt.step()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        sec = (time.time() - t0) / steps
        out = {"ok": True, "sec_per_micro": sec, "tokens_per_s": micro * L / sec,
               "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30, "params": params}
    except torch.OutOfMemoryError:
        pass
    del model, fwd, opt
    gc.collect()
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=10, help="micro-steps timed after warm-up")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--write-overrides", default=None,
                    help="file to receive the train.py overrides (used by scripts/run_e06.sh)")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    t, g = cfg["train"], cfg.get("guard", {})
    torch.backends.cuda.matmul.allow_tf32 = True
    L = t["seq_len"]
    tokens_per_step = L * t["micro_batch"] * t["grad_accum"]

    candidates, m = [], t["micro_batch"]
    while m >= 1:
        candidates.append((m, False))
        m //= 2
    candidates += [(m, True) for m in (4, 2, 1)]
    tried, best = [], None
    for micro, ckpt in candidates:
        if tokens_per_step % (micro * L):
            continue
        r = _try(cfg, micro, ckpt, a.steps, a.warmup)
        tried.append({"micro_batch": micro, "grad_checkpointing": ckpt,
                      "result": "fits" if r["ok"] else "OOM"})
        print(tried[-1], flush=True)
        if r["ok"]:
            best = (micro, ckpt, r)
            break
    if best is None:
        raise SystemExit("no configuration fits on this GPU even with checkpointing at micro 1")

    micro, ckpt, r = best
    hours = t["train_tokens"] / r["tokens_per_s"] / 3600
    price = g.get("usd_per_hour", 0.0)
    accum = tokens_per_step // (micro * L)
    overrides = [f"train.micro_batch={micro}", f"train.grad_accum={accum}"]
    if ckpt:
        overrides.append("train.grad_checkpointing=true")
    rep = {"arch": cfg["model"]["arch"], "params": r["params"],
           "gpu": torch.cuda.get_device_name(0), "tokens_per_step": tokens_per_step,
           "micro_batch": micro, "grad_accum": accum, "grad_checkpointing": ckpt,
           "tokens_per_s": round(r["tokens_per_s"]), "peak_mem_gib": round(r["peak_mem_gib"], 2),
           "projected_train_hours": round(hours, 2),
           "projected_train_usd": round(hours * price, 2), "usd_per_hour": price,
           "cost_guard_hours": g.get("max_projected_hours"),
           "train_overrides": " ".join(overrides), "search": tried,
           "note": "random tokens; excludes eval/checkpoint time; add 20-30% contingency"}
    print(json.dumps(rep, indent=2))
    if a.write_overrides:
        with open(a.write_overrides, "w") as f:
            f.write(" ".join(overrides) + "\n")


if __name__ == "__main__":
    main()
