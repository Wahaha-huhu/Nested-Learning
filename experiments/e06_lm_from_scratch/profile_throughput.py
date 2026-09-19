"""Measure throughput / peak memory of an E06 arm on random tokens, BEFORE committing to a run.

For each mode (no checkpointing, block checkpointing; optionally torch.compile) it finds the
largest micro-batch that fits, measures it, and picks the FASTEST setting. Hope's chunk loop is
launch-bound, so a larger micro-batch with checkpointing usually beats a small one without.
Tokens per optimiser step stay fixed via grad_accum, so the optimisation is identical. Prints the projected hours/USD for the full
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


def _try(cfg, micro, ckpt, steps, warmup, compile_=False):
    model = build_model(cfg["model"]).cuda()
    model.grad_checkpointing = ckpt
    fwd = torch.compile(model) if compile_ else model
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    V, L = cfg["model"]["vocab_size"], cfg["train"]["seq_len"]
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
    ap.add_argument("--try-compile", action="store_true",
                    help="also measure torch.compile variants (slow to compile for Hope)")
    ap.add_argument("--max-micro", type=int, default=32)
    ap.add_argument("--write-overrides", default=None,
                    help="file to receive the train.py overrides (used by scripts/run_e06.sh)")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    t, g = cfg["train"], cfg.get("guard", {})
    torch.backends.cuda.matmul.allow_tf32 = True
    L = t["seq_len"]
    tokens_per_step = L * t["micro_batch"] * t["grad_accum"]

    micros = [m for m in (32, 16, 8, 4, 2, 1)
              if m <= a.max_micro and tokens_per_step % (m * L) == 0]
    compiles = sorted({bool(t.get("compile")), True} if a.try_compile else {bool(t.get("compile"))})
    tried, results = [], []
    for comp in compiles:
        for ckpt in (False, True):
            for micro in micros:                     # largest that fits for this mode
                r = _try(cfg, micro, ckpt, a.steps, a.warmup, comp)
                entry = {"micro_batch": micro, "grad_checkpointing": ckpt, "compile": comp,
                         "result": "OOM" if not r["ok"] else f"{r['tokens_per_s']:.0f} tok/s, "
                                   f"{r['peak_mem_gib']:.1f} GiB"}
                tried.append(entry)
                print(entry, flush=True)
                if r["ok"]:
                    results.append((r["tokens_per_s"], micro, ckpt, comp, r))
                    break
    if not results:
        raise SystemExit("no configuration fits on this GPU even with checkpointing at micro 1")
    _, micro, ckpt, comp, r = max(results, key=lambda x: x[0])
    best = (micro, ckpt, r)

    micro, ckpt, r = best
    hours = t["train_tokens"] / r["tokens_per_s"] / 3600
    price = g.get("usd_per_hour", 0.0)
    accum = tokens_per_step // (micro * L)
    overrides = [f"train.micro_batch={micro}", f"train.grad_accum={accum}"]
    overrides.append(f"train.grad_checkpointing={'true' if ckpt else 'false'}")
    overrides.append(f"train.compile={'true' if comp else 'false'}")
    rep = {"arch": cfg["model"]["arch"], "params": r["params"],
           "gpu": torch.cuda.get_device_name(0), "tokens_per_step": tokens_per_step,
           "micro_batch": micro, "grad_accum": accum, "grad_checkpointing": ckpt,
           "compile": comp, "tokens_per_s": round(r["tokens_per_s"]), "peak_mem_gib": round(r["peak_mem_gib"], 2),
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
