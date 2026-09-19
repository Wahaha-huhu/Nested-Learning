# Cost and feasibility

Prices: Runpod list prices checked 2026-09-19 (per the feasibility review): **RTX 4090 24GB
$0.74/h**, **A100 80GB $1.59/h**. Throughput figures below are *assumptions* (4090: 40-100
effective TFLOP/s; A100: 80-160) until `make e06-profile` replaces them with measurements.
Regenerate the tables with `make cost` (`nlrepro/cost/estimator.py`).

## Is the paper's scale reachable?

| Paper workload | Single-GPU estimate (dense, overhead 1x) | Verdict |
|---|---|---|
| Fig. 6: Llama-3 3B/8B + 15B continued-pretraining tokens | 3B: 470-940 A100-h ($745-1,490); 8B state (119 GiB) does not fit one GPU | Out of scope |
| Table 2: 760M / 30B tokens (from scratch) | 240-475 A100-h ($380-755) per arm, x2-5 for a naive Hope | Out of scope |
| Table 2: 1.3B / 100B tokens | 1,350-2,700 A100-h ($2.2-4.3K) per arm, x2-5 for Hope | Out of scope |

Training 3B/8B from scratch is several orders of magnitude beyond a proof-of-concept budget.
Smaller models can test the *mechanisms* (online multi-frequency updates, self-modifying
memories, relative improvement over a matched baseline); they cannot test paper-level scores.

## What we run now (all on one RTX 4090)

| ID | Workload | Estimate | Why this size |
|---|---|---|---|
| E00 | 2-D toy problem, CPU | $0 (seconds; already run) | Exact problem from the paper |
| E01 stage A | Qwen2.5-0.5B, ~60K-token stream per dataset, 3 arms x 3 datasets, no continued pretraining | ~1 GPU-h total, **< $1** | Smallest capable Apache-2.0 decoder; the CMS mechanism is exercised at test time |
| E06 Transformer++ | 109.5M params, 1.0B tokens, seq 2048 | 2.1-5.4 h, **$2-4** | ~9 tokens/param; enough for a relative comparison |
| E06 Hope | 110.6M params, 1.0B tokens | x2: 3.7-9.2 h ($3-7); x5: 9-23 h ($7-17) | Overhead unknown until profiled; cost guard caps it at 30 h |
| E06 data | 1.05B tokens FineWeb-Edu tokenisation | CPU, ~0.5-1.5 h of pod time ($0.4-1.1) | Shared by both arms |
| E06 eval | val ppl + 9 lm-eval tasks, both arms | ~0.5 GPU-h ($0.4) | |

**Planned total for this round: about $10-30**, dominated by the Hope arm's unknown overhead.
Add 20-30% operational contingency after profiling. Storage: ~2 GB tokens + ~2 GB checkpoints +
~5 GB HF cache; a 20-50 GB network volume costs $1.4-3.5/month.

## A100 vs 4090

Every current workload fits in 24 GB. E01: 0.5B fp32 master weights + AdamW state for its MLPs
is ~6 GB. E06: the first profile OOMed at micro-batch 16 because the loss kept ~4 GiB fp32 copies
of the (16 x 2048) x 32K logits; the loss is now chunked, and the profiler picks the largest
micro-batch that fits while keeping 262,144 tokens per optimiser step (grad_accum compensates, so
the optimisation is unchanged). Use the **4090**. Switch to
an A100 80GB only for E01 with a 3B backbone (paper-nearer; ~45 GiB state if fully trained) or
E06 above ~350M params.

## Measure before committing

1. `make smoke` (~15 min, < $0.25): runs every pipeline end-to-end and ends with a full-size
   throughput profile of both E06 arms (tokens/s, peak memory, projected hours and USD).
2. The trainer re-projects wall time from its first 30 optimiser steps and aborts the arm if it
   exceeds `guard.max_projected_hours` (Transformer 12 h, Hope 30 h). Raise it explicitly if the
   profile justifies it: `... train --config hope_110m.yaml guard.max_projected_hours=40`.

## Why no seed or hyperparameter sweeps

The request is a proof of concept of the effect, so every run uses one seed and hyperparameters
fixed a priori from standard practice (listed in the configs, justified in EXPERIMENT_PLAN.md).
Consequence: small differences (e.g. < 1 ppl or < 2 accuracy points) are not distinguishable from
seed noise and should be reported as inconclusive, not as wins.
