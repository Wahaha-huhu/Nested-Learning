# Nested Learning — proof-of-concept reproduction

Small-scale reproduction of *Nested Learning: The Illusion of Deep Learning Architectures*
(Behrouz, Razaviyayn, Zhong, Mirrokni; arXiv:2512.24695), following the paper's experiment order
and designed for a single RTX 4090 on Runpod.

**Goal:** test whether the paper's mechanisms (delta momentum, the Continuum Memory System,
Hope) show their claimed *effects* at affordable scale, not to match paper scores.
No official code exists; every reconstruction choice is listed in `docs/DEVIATIONS.md`.

## Status

| ID | Paper | Status | Cost |
|---|---|---|---|
| E00 | Fig. 4 — standard vs delta momentum | **done** (CPU): qualitative picture reproduced, speed claim not | $0 |
| E01 | Fig. 6 — class-incremental CLINC / Banking / DBpedia, ICL vs CMS | **ready**, stage A | < $1 |
| E06 | Table 2 — Hope vs Transformer++ trained from scratch (110M, 1B tokens) | **ready** | ~$10-30 |
| E02-E05, E07-E13 | remaining experiments | planned | see docs |

## Quick start (Runpod, see `docs/RUNPOD.md`)

```bash
bash scripts/runpod_setup.sh   # pinned deps + unit tests
make smoke                     # ~15 min: every pipeline end-to-end + measured E06 throughput
make e01                       # E01 stage A, ~1 GPU-hour
make e06                       # tokenise, train and evaluate both E06 arms (resumable, use tmux)
```
Locally, `make e00` and `make test` run on CPU (torch tests are skipped without torch).

## Layout

```
nlrepro/                 library
  optim/toy_momentum.py    E00 objective + standard/delta momentum (numpy, float64)
  models/transformer.py    Transformer++ baseline (RoPE, SwiGLU, RMSNorm)
  models/titans_selfmod.py self-modifying Titans, chunk-parallel + reference loop
  models/cms.py            Continuum Memory System levels (in-context + static)
  models/hope.py           Hope = self-modifying Titans + CMS
  models/hf_cms.py         Sec. 7.3 level stacking on a pretrained HF decoder (E01)
  data/                    FineWeb-Edu token pipeline, class-incremental datasets
  train/lm_trainer.py      trainer with resume and a cost guard
  eval/                    lm-eval-harness wrapper, cached label scoring
  cost/estimator.py        planning arithmetic (`make cost`)
experiments/eXX_*/       one folder per paper experiment: configs, entry points, README
scripts/                 Runpod setup and run scripts
tests/                   numpy algebra checks (run anywhere) + torch tests (run on the pod)
docs/                    cost & feasibility, experiment plan, deviations, Runpod guide
```

## Verification status

The E00 code, cost estimator and the closed-form algebra of the Titans and CMS updates
(chunked form vs token-by-token loop, analytic vs finite-difference gradients) were tested
without a GPU. The PyTorch model, training and evaluation code has **not yet been executed**;
`scripts/runpod_setup.sh` runs its unit tests and `make smoke` exercises every pipeline before
any budgeted run.
