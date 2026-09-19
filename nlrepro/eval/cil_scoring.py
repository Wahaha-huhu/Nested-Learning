"""Label scoring for E01: argmax over seen classes of the mean per-token log-probability of
" <label>" after "<stream context><Query: ...\\nIntent:>" (D-E01-5: length-normalised).

The stream context (~1.5K tokens) is shared by every query at an evaluation point, so it is
encoded once; each query extends it once; the candidate labels are then scored in a batch.
Written against transformers 4.46 (DynamicCache). `self_check` compares with uncached scoring
in fp32 (so it tests the caching logic, not bf16 rounding) at the first evaluation of every arm.

Precision: scoring runs in fp32 by default. Under bf16 autocast, logits of magnitude ~20-30 are
quantised in steps of ~0.125, and cached vs uncached kernels round differently (observed max
difference 0.169 nats), which is enough to flip close label decisions.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _to_legacy(pkv):
    return pkv.to_legacy_cache() if hasattr(pkv, "to_legacy_cache") else pkv


def _cache(legacy, batch: int):
    from transformers import DynamicCache
    return DynamicCache.from_legacy_cache(tuple(
        (k.expand(batch, -1, -1, -1), v.expand(batch, -1, -1, -1)) for k, v in legacy))


class LabelScorer:
    def __init__(self, model, tokenizer, device, cand_batch: int = 64, dtype: str = "fp32"):
        assert dtype in ("fp32", "bf16")
        self.model, self.tok, self.dev, self.cb = model, tokenizer, device, cand_batch
        self.bf16 = dtype == "bf16" and device == "cuda"
        self.pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self._label_ids: dict[str, list[int]] = {}

    def label_ids(self, label: str) -> list[int]:
        if label not in self._label_ids:
            self._label_ids[label] = self.tok(" " + label, add_special_tokens=False)["input_ids"]
        return self._label_ids[label]

    def _fwd(self, ids, cache=None):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.bf16):
            return self.model(input_ids=ids, past_key_values=cache, use_cache=True)

    @torch.no_grad()
    def encode_context(self, ctx_ids: list[int]):
        if not ctx_ids:
            return None
        out = self._fwd(torch.tensor([ctx_ids], device=self.dev))
        return _to_legacy(out.past_key_values)

    @torch.no_grad()
    def score(self, ctx_legacy, query_ids: list[int], labels: list[str]) -> torch.Tensor:
        cache = _cache(ctx_legacy, 1) if ctx_legacy is not None else None
        out = self._fwd(torch.tensor([query_ids], device=self.dev), cache)
        first_lp = F.log_softmax(out.logits[0, -1].float(), -1)
        q_legacy = _to_legacy(out.past_key_values)
        scores = []
        for i in range(0, len(labels), self.cb):
            labs = [self.label_ids(l) for l in labels[i:i + self.cb]]
            n, T = len(labs), max(len(x) for x in labs)
            x = torch.full((n, T), self.pad, dtype=torch.long, device=self.dev)
            for j, ids in enumerate(labs):
                x[j, :len(ids)] = torch.tensor(ids, device=self.dev)
            lp = torch.zeros(n, device=self.dev)
            for j, ids in enumerate(labs):
                lp[j] = first_lp[ids[0]]
            if T > 1:
                logits = self._fwd(x, _cache(q_legacy, n)).logits.float()   # right padding ok
                logz = torch.logsumexp(logits[:, :-1], -1)                    # (n, T-1)
                tok = logits[:, :-1].gather(-1, x[:, 1:, None]).squeeze(-1) - logz
                mask = torch.zeros(n, T - 1, device=self.dev)
                for j, ids in enumerate(labs):
                    mask[j, :len(ids) - 1] = 1
                lp = lp + (tok * mask).sum(-1)
            lens = torch.tensor([len(ids) for ids in labs], device=self.dev, dtype=torch.float)
            scores.append(lp / lens)
        return torch.cat(scores)

    @torch.no_grad()
    def score_uncached(self, ctx_ids, query_ids, labels) -> torch.Tensor:
        res = []
        for l in labels:
            lab = self.label_ids(l)
            ids = torch.tensor([ctx_ids + query_ids + lab], device=self.dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.bf16):
                logits = self.model(input_ids=ids).logits[0].float()
            lp = F.log_softmax(logits[-len(lab) - 1:-1], -1)
            res.append(lp.gather(-1, torch.tensor(lab, device=self.dev)[:, None]).mean())
        return torch.stack(res)

    def self_check(self, ctx_ids, query_ids, labels, tol: float = 0.02) -> float:
        labels = labels[:4]
        saved, self.bf16 = self.bf16, False          # compare in fp32: logic, not rounding
        try:
            a = self.score(self.encode_context(ctx_ids), query_ids, labels)
            b = self.score_uncached(ctx_ids, query_ids, labels)
        finally:
            self.bf16 = saved
        err = float((a - b).abs().max())
        if err > tol:
            raise RuntimeError(f"cached scoring disagrees with uncached scoring (max err {err:.3f})"
                               " in fp32 - a logic error; check the transformers version (4.46.x)")
        return err
