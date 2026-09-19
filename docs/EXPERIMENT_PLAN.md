# Experiment plan (paper order)

Order follows the first experimental discussion in the paper (arXiv:2512.24695v1), as in the
feasibility review. Status: **done** / **ready** (code complete, awaiting GPU run) / planned.

| ID | Paper | Status | This round |
|---|---|---|---|
| E00 | Sec. 4.4, Fig. 4: standard vs delta momentum | **done** (CPU) | Result below |
| E01 | Sec. 9.1, Fig. 6: class-incremental CLINC / Banking / DBpedia | **ready** | Stage A |
| E02 | Sec. 9.1, Fig. 7: levels vs in-context QA, MK-NIAH | planned | reuses E01 adapter |
| E03 | Sec. 9.1, Fig. 8: Kalamang / Manchu translation | planned | needs a more capable backbone |
| E04-E05 | Sec. 9.2: RULER, BABILong | planned | reuse E06 checkpoints |
| E06 | Sec. 9.3, Table 2: LM + reasoning, from scratch | **ready** | 110M / 1B tokens, 2 arms |
| E07-E13 | Secs. 9.4-9.7 | planned | see the feasibility review |

## E00: result (reconstruction, not the authors' constants)

The paper gives only psi(r, theta) = r^2 + k (r - theta + a sin(w r))^2 and the start (-3.5, 2).
Constants fixed before running: k = 10, a = 1, w = 2, eta = 0.004, momentum alpha = 0.9,
converged when psi < 1e-6 for 10 steps. The printed rule `m(alpha - g^T g)` has no usable scale,
so two conventions are reported (D-E00-2):

| Optimiser | Steps to converge |
|---|---:|
| Standard momentum | 670 |
| Delta momentum, "normalized" key (beta = 0.5) | 4,556 |
| Delta momentum, "gd_l2" (decay c = min(eta, alpha/‖g‖²)) | 719 |

Robustness report over a declared grid (not used to choose anything): "gd_l2" converges faster
than standard momentum in 89 / 162 settings (median step ratio 1.02); "normalized" in 3 / 486.
**Reading:** qualitatively the paper's picture holds (standard momentum overshoots far past the
basin; both delta variants take a more direct path, see `results/figure4_reproduction.png`), but
the quantitative "delta converges faster" claim is **not reproduced** at our constants: at best
the gradient-magnitude-dependent variant is on par. Artefacts:
`experiments/e00_delta_momentum/results/`.

## E01 stage A: class-incremental learning with CMS (ready)

* **Backbone:** Qwen2.5-0.5B base (Apache-2.0). The paper used Llama-3 3B/8B. The same config runs
  `Qwen/Qwen2.5-1.5B` or `meta-llama/Llama-3.2-3B` (gated, A100) by changing `model_id`.
* **Protocol:** classes shuffled with seed 0 and split into tasks (CLINC 10 x 15, Banking 7 x 11,
  DBpedia 7 x 10). The model reads one stream of `Query: ...\nIntent: <label>` examples, 20 per
  class, task after task. After each task it is tested on held-out queries of all seen classes,
  candidates = seen labels, prediction = highest length-normalised label log-probability.
* **Arms (same stream, same 1,536-token test-time context, same scorer):**
  `icl` (frozen, the paper's ICL baseline); `cms_1lvl` (all MLPs updated every 512 tokens: single
  frequency control, our addition); `cms_3lvl` (MLPs split into levels updated every 512 / 2,048 /
  8,192 tokens: Sec. 7.3 ad-hoc level stacking).
* **Fixed choices and why:** inner AdamW lr 2e-5 (the usual LLM fine-tuning LR; not tuned);
  chunk sizes 512 / 2K / 8K are the schedule labels of the paper's Fig. 7; 20 train examples per
  class keeps the stream (~60K tokens) far beyond the attention window, which is where ICL
  should forget.
* **Expected signal:** the paper predicts `cms_3lvl` > `icl` in final accuracy with less
  forgetting. Our addition tests whether multiple frequencies beat a single frequency.
* **Not done in stage A:** the paper's 15B-token continued pretraining with the adapted MLPs.
  Stage B (a 100M-token pilot) is worth doing only if stage A shows a signal.
* EWC and InCA baselines are not run in stage A (paper numbers for them are from prior work).

## E06: Hope vs Transformer++ from scratch (ready)

* **Paper setup:** 760M/30B and 1.3B/100B tokens, FineWeb-Edu + long documents, 32K vocab, AdamW
  with per-model tuned LR, zero-shot Wiki/LMB/PIQA/Hella/Wino/ARC-e/ARC-c/SIQA/BoolQ.
* **Ours:** 12 layers, d = 768, 12 heads, both ~110M params (+1.0% for Hope), 1.0B tokens of
  FineWeb-Edu sample-10BT, Llama-2 32K tokenizer, seq 2048, 262K tokens/step (3,815 steps),
  AdamW lr 1e-3, betas (0.9, 0.95), wd 0.1, 2% warmup, cosine to 10%, clip 1.0, bf16, seed 0,
  identical data order for both arms. One shared LR instead of per-model tuning (D-E06-3).
* **Hope config:** self-modifying Titans (chunk 64, width-4 conv, adaptive k/v memories with
  self-generated values) + CMS [in-context MLP memory updated every 512 tokens -> static SwiGLU].
* **Evaluation:** held-out FineWeb-Edu perplexity (primary, most sensitive at this scale), then
  lm-eval zero-shot on the paper's tasks. Expect reasoning accuracies near chance for both arms.
* **Expected signal:** the paper reports Hope ahead of Transformer++ (lower perplexity, higher average accuracy) at 760M and 1.3B.
  At 110M/1B, a recurrent model typically trails a Transformer; parity or better would be the
  interesting outcome, and a clear loss is a legitimate negative result for this reconstruction.
