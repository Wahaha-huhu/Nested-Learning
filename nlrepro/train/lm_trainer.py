"""Single-GPU language-model trainer shared by both E06 arms.

Fairness: both arms use the same data sampler seed, token budget, optimiser, schedule and
hyperparameters. Nothing is tuned (docs/EXPERIMENT_PLAN.md, E06).

Cost guard: after `guard.profile_steps` optimiser steps the trainer projects total wall time
from measured step times and aborts if it exceeds `guard.max_projected_hours`. `guard.max_wall_hours`
is a hard stop that checkpoints and exits cleanly (re-running the same command resumes).
"""
from __future__ import annotations

import math
import os
import time

import torch
import torch.nn.functional as F

from nlrepro.data.tokens import TokenSampler, fixed_val_batches
from nlrepro.models import build_model, count_params
from nlrepro.utils.common import JsonlLogger, environment, seed_everything, write_json


def lr_at(step: int, total: int, peak: float, warmup: int, min_ratio: float) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p)))


def param_groups(model, weight_decay: float):
    decay, no_decay, seen = [], [], set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        # no decay: norms, biases, embeddings, meta-learned memory initial states
        if p.ndim < 2 or "embed" in name or name.endswith(("M0", "W1_0", "W2_0")):
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


@torch.no_grad()
def evaluate_val(model, val_path, seq_len, n_seqs, batch_size, device, dtype) -> float:
    model.eval()
    tot, n = 0.0, 0
    for x, y in fixed_val_batches(val_path, seq_len, n_seqs, batch_size, device):
        with torch.autocast("cuda", dtype=dtype, enabled=device.startswith("cuda")):
            logits = model(x)
        tot += F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1),
                               reduction="sum").item()
        n += y.numel()
    model.train()
    return tot / max(1, n)


def train(cfg: dict) -> dict:
    run_dir = cfg["run_dir"]
    os.makedirs(run_dir, exist_ok=True)
    t_cfg, d_cfg, g_cfg = cfg["train"], cfg["data"], cfg.get("guard", {})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    seed_everything(t_cfg.get("seed", 0))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = build_model(cfg["model"]).to(device)
    model.grad_checkpointing = t_cfg.get("grad_checkpointing", False)
    n_params = count_params(model)
    fwd = torch.compile(model) if t_cfg.get("compile", False) else model

    seq_len, micro_bs, accum = t_cfg["seq_len"], t_cfg["micro_batch"], t_cfg["grad_accum"]
    tokens_per_step = seq_len * micro_bs * accum
    total_steps = math.ceil(t_cfg["train_tokens"] / tokens_per_step)
    warmup = max(1, int(t_cfg.get("warmup_frac", 0.02) * total_steps))

    opt = torch.optim.AdamW(param_groups(model, t_cfg["weight_decay"]), lr=t_cfg["lr"],
                            betas=tuple(t_cfg.get("betas", (0.9, 0.95))), eps=1e-8,
                            fused=device == "cuda")
    sampler = TokenSampler(os.path.join(d_cfg["dir"], "train.bin"), seq_len, micro_bs,
                           t_cfg.get("data_seed", 1234), device)
    log = JsonlLogger(os.path.join(run_dir, "train_log.jsonl"))

    ckpt_path = os.path.join(run_dir, "checkpoint.pt")
    step, elapsed_prev = 0, 0.0
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sampler.load_state_dict(ck["sampler"])
        step, elapsed_prev = ck["step"], ck.get("elapsed", 0.0)
        print(f"resumed from step {step}")

    def save(tag: str = "checkpoint.pt"):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sampler": sampler.state_dict(), "step": step, "cfg": cfg,
                    "elapsed": elapsed_prev + (time.time() - t_start)},
                   os.path.join(run_dir, tag + ".tmp"))
        os.replace(os.path.join(run_dir, tag + ".tmp"), os.path.join(run_dir, tag))

    write_json(os.path.join(run_dir, "run_info.json"),
               {"config": cfg, "params": n_params, "tokens_per_step": tokens_per_step,
                "total_steps": total_steps, "env": environment()})
    print(f"params={n_params} steps={total_steps} tokens/step={tokens_per_step:,}")

    model.train()
    t_start, t_mark, step_times = time.time(), time.time(), []
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    status = "completed"
    while step < total_steps:
        lr = lr_at(step, total_steps, t_cfg["lr"], warmup, t_cfg.get("min_lr_ratio", 0.1))
        for g in opt.param_groups:
            g["lr"] = lr
        t0 = time.time()
        loss_acc = 0.0
        for _ in range(accum):
            x, y = sampler.next()
            with torch.autocast("cuda", dtype=dtype, enabled=device == "cuda"):
                logits = fwd(x)
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1))
            (loss / accum).backward()
            loss_acc += loss.item() / accum
        if not math.isfinite(loss_acc):
            status = "diverged"
            log.log(step=step, event="non_finite_loss", loss=loss_acc)
            break
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), t_cfg.get("grad_clip", 1.0))
        opt.step()
        opt.zero_grad(set_to_none=True)
        if device == "cuda":
            torch.cuda.synchronize()
        step_times.append(time.time() - t0)
        step += 1

        if step % t_cfg.get("log_every", 10) == 0:
            dt = time.time() - t_mark
            t_mark = time.time()
            log.log(step=step, loss=loss_acc, lr=lr, grad_norm=float(gnorm),
                    tok_per_s=t_cfg.get("log_every", 10) * tokens_per_step / dt,
                    peak_mem_gib=(torch.cuda.max_memory_allocated() / 2**30
                                  if device == "cuda" else 0.0))

        n_prof = g_cfg.get("profile_steps", 30)
        if len(step_times) == n_prof and g_cfg.get("max_projected_hours"):
            per = sum(step_times[5:]) / max(1, n_prof - 5)       # skip warm-up/compile steps
            proj = per * (total_steps - step) / 3600 + elapsed_prev / 3600
            price = g_cfg.get("usd_per_hour", 0.0)
            log.log(step=step, event="projection", sec_per_step=per, projected_hours=proj,
                    projected_usd=proj * price)
            if proj > g_cfg["max_projected_hours"]:
                status = "aborted_by_cost_guard"
                log.log(step=step, event=status, limit_hours=g_cfg["max_projected_hours"])
                break

        if step % t_cfg.get("eval_every", 500) == 0 or step == total_steps:
            vl = evaluate_val(model, os.path.join(d_cfg["dir"], "val.bin"), seq_len,
                              t_cfg.get("val_seqs", 256), micro_bs, device, dtype)
            log.log(step=step, event="val", val_loss=vl, val_ppl=math.exp(vl))
        if step % t_cfg.get("ckpt_every", 500) == 0:
            save()
        if g_cfg.get("max_wall_hours") and (time.time() - t_start) / 3600 > g_cfg["max_wall_hours"]:
            status = "stopped_wall_clock"
            break

    save()
    if status == "completed":
        torch.save({"model": model.state_dict(), "cfg": cfg}, os.path.join(run_dir, "final.pt"))
    summary = {"status": status, "steps": step, "total_steps": total_steps,
               "tokens": step * tokens_per_step, "params": n_params,
               "wall_hours": (elapsed_prev + time.time() - t_start) / 3600,
               "mean_sec_per_step": sum(step_times) / max(1, len(step_times)),
               "peak_mem_gib": (torch.cuda.max_memory_allocated() / 2**30
                                if device == "cuda" else 0.0)}
    write_json(os.path.join(run_dir, "train_summary.json"), summary)
    return summary
