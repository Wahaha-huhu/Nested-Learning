# Deviations and reconstruction choices

No official Hope code or checkpoints were found (feasibility review). Wherever the paper is
silent, ambiguous or printed inconsistently, the choice below was made **before** running and is
not presented as the authors' algorithm. IDs are referenced from code comments.

## E00 (Sec. 4.4, Fig. 4)

| ID | Paper | Ours | Reason |
|---|---|---|---|
| D-E00-1 | psi and start (-3.5, 2) only | k=10, a=1, w=2, eta=0.004, alpha=0.9, tol 1e-6 x 10 steps | constants not given |
| D-E00-2 | m_{i+1} = m_i(alpha - ∇ᵀ∇) - eta P ∇ | (a) "normalized": (alpha I - beta ĝĝᵀ) m, beta=0.5; (b) "gd_l2": (alpha I - c ggᵀ) m, c = min(eta, alpha/‖g‖²); P = I | the literal form has no scale and diverges (‖g‖≈340 at the start); both readings reported |

The "normalized" convention was run first; after it lost, "gd_l2" was added because it keeps the
gradient-magnitude dependence the paper's explanation relies on. Both are reported.

## Self-modifying Titans (Sec. 8.1-8.2) — `nlrepro/models/titans_selfmod.py`

| ID | Paper | Ours |
|---|---|---|
| D-HOPE-1 | 2-layer residual MLP memories | matrix-valued memories (the paper's own "simplest case", Eq. 92); allows an exact intra-chunk dual form |
| D-HOPE-2 | Eq. 92 prints `- eta v_hat kᵀ` | `+ eta v_hat kᵀ` (L2/delta-rule sign); the printed sign makes the memory move away from its targets |
| D-HOPE-3 | retention gate alpha | decay toward the meta-learned initial state M0 (state = M0 + D, D decays) |
| D-HOPE-4 | memories for k, v, q, eta, alpha, memory | k, v, memory are in-context memories; q, eta, alpha come from static projections (data-dependent) |
| D-HOPE-5 | self-generated values for all memories | M_k, M_v self-generated; M_memory uses the standard value v (`self_values_memory=true` switches) |
| D-HOPE-6 | chunk-wise parallel training | mini-batch semantics: errors at chunk-start state; memory read exact within the chunk |
| D-HOPE-7 | two chunk sizes | one size (64) by default; `proj_every` sets a lower frequency for M_k, M_v. The paper's figure suggests 16; 64 is used for throughput |
| D-HOPE-9 | self-generated values v_hat = M v | the self-referential error of M_k, M_v uses l2norm(v); with raw v = M_v u the update is quadratic in M_v and diverges (found in the first smoke run) |
| D-HOPE-8 | inner step sizes unspecified | eta_t = 0.2 sigmoid(.), M_k/M_v step x0.1, retention init sigmoid(5)=0.993 |

## CMS in the from-scratch model (Sec. 7.1) — `nlrepro/models/cms.py`

| ID | Paper | Ours |
|---|---|---|
| D-CMS-1 | Eq. 71 error term = task-loss (NTP) gradient | local associative L2 loss ‖M(k) - v‖² (the NTP gradient is not available inside the forward pass) |
| D-CMS-2 | chain of MLP levels, schedules not given for Table 2 | 2 levels: in-context 2-layer residual MLP memory (C = 512 tokens, reset per sequence to a meta-learned init) -> static SwiGLU (frequency 0); per-level norm + residual |
| D-CMS-4 | retention per update | one gate per chunk update = exp(mean of the per-token log-gates) |
| D-CMS-3 | parameter budget per level unspecified | static SwiGLU hidden 1152 so total params match Transformer++ within 1% |

## E01 (Sec. 7.3, 9.1) — `nlrepro/models/hf_cms.py`, `experiments/e01_class_incremental`

| ID | Paper | Ours |
|---|---|---|
| D-E01-1 | "initialise CMS levels with pretrained MLP blocks" | each decoder layer's MLP is a CMS block, layer j -> level j mod 3 (exactly the pretrained model at init) |
| D-E01-2 | Eq. 71 with an arbitrary optimiser | one AdamW per level, lr 2e-5, fp32 master weights, NTP loss on the stream |
| D-E01-3 | Llama-3 3B/8B + 15B continued-pretraining tokens | Qwen2.5-0.5B, no continued pretraining (stage A) |
| D-E01-4 | InCA protocol, splits unpublished | own class order (seed 0), 20 train / 5-10 test per class, tasks of 10-15 classes |
| D-E01-5 | prediction method unspecified | argmax over seen labels of mean per-token label log-prob, context = last 1,536 stream tokens |
| D-E01-6 | baselines ICL, EWC, InCA | ICL + single-frequency control; EWC/InCA not run |

## E06 (Sec. 9.3)

| ID | Paper | Ours |
|---|---|---|
| D-E06-1 | 760M/30B, 1.3B/100B | ~110M / 1.0B tokens |
| D-E06-2 | FineWeb-Edu + long-context documents | FineWeb-Edu sample-10BT only |
| D-E06-3 | per-model tuned LR | one shared LR (1e-3), no tuning, one seed |
| D-E06-4 | baselines incl. RetNet, DeltaNet, RWKV-7, Titans, Samba ... | Transformer++ only |
