"""Zero-shot evaluation of E06 checkpoints with lm-evaluation-harness (paper Table 2 tasks).

Paper tasks: Wikitext ppl, LAMBADA, PIQA, HellaSwag, WinoGrande, ARC-e, ARC-c, SIQA, BoolQ.
At ~110M params / 1B tokens most reasoning tasks will sit near chance for both arms; the
informative signals are validation/Wikitext perplexity, LAMBADA, and the arm difference.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    from lm_eval.api.model import LM
except ImportError:  # keeps the module importable without lm_eval
    LM = object


class NLReproLM(LM):
    def __init__(self, model, tokenizer, device="cuda", max_len=2048, batch_size=16):
        super().__init__()
        self.model, self.tok = model.eval(), tokenizer
        self.dev, self.max_len, self.bs = device, max_len, batch_size
        self.bos = tokenizer.eos_token_id    # documents are EOS-separated in training data

    def _enc(self, s: str) -> list[int]:
        return self.tok(s, add_special_tokens=False)["input_ids"] if s else []

    @torch.no_grad()
    def _score(self, pairs):
        """pairs: list of (context_ids, continuation_ids) -> list of (logprob_sum, is_greedy)."""
        out = []
        for i in range(0, len(pairs), self.bs):
            batch = pairs[i:i + self.bs]
            seqs, spans = [], []
            for ctx, cont in batch:
                ids = ([self.bos] + ctx + cont)[-(self.max_len + 1):]
                seqs.append(ids)
                spans.append(len(cont))
            L = max(len(s) for s in seqs) - 1
            x = torch.full((len(seqs), L), self.bos, dtype=torch.long)
            for j, s in enumerate(seqs):               # right padding is harmless (causal)
                x[j, :len(s) - 1] = torch.tensor(s[:-1])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.dev == "cuda"):
                logits = self.model(x.to(self.dev))
            for j, (s, n) in enumerate(zip(seqs, spans)):   # upcast only the scored rows
                tgt = torch.tensor(s[-n:], device=self.dev)
                rows = F.log_softmax(logits[j, len(s) - 1 - n:len(s) - 1].float(), -1)
                out.append((rows.gather(-1, tgt[:, None]).sum().item(),
                            bool((rows.argmax(-1) == tgt).all().item())))
        return out

    def loglikelihood(self, requests):
        pairs = [(self._enc(r.args[0]), self._enc(r.args[1])) for r in requests]
        return self._score(pairs)

    def loglikelihood_rolling(self, requests):
        from lm_eval.utils import get_rolling_token_windows, make_disjoint_window
        res = []
        for r in requests:
            windows = [make_disjoint_window(w) for w in get_rolling_token_windows(
                token_list=self._enc(r.args[0]), prefix_token=self.bos,
                max_seq_len=self.max_len, context_len=1)]
            # windows already include the prefix token as context; drop our extra BOS
            pairs = [(ctx[1:] if ctx and ctx[0] == self.bos else ctx, cont)
                     for ctx, cont in windows]
            res.append(sum(s for s, _ in self._score(pairs)))
        return res

    def generate_until(self, requests):
        raise NotImplementedError("generation tasks are not part of the E06 suite")


def run_harness(model, tokenizer, tasks, device, batch_size=16, limit=None) -> dict:
    """Runs each task separately so one broken dataset does not sink the whole evaluation."""
    import lm_eval
    lm = NLReproLM(model, tokenizer, device=device, batch_size=batch_size)
    results = {}
    for t in tasks:
        try:
            r = lm_eval.simple_evaluate(model=lm, tasks=[t], num_fewshot=0, limit=limit,
                                        log_samples=False)
            results[t] = r["results"].get(t, r["results"])
        except Exception as e:  # recorded, not hidden
            results[t] = {"error": f"{type(e).__name__}: {e}"}
    return results
