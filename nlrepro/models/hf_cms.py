"""Ad-hoc level stacking for a pretrained decoder (paper Sec. 7.3), used by E01.

Paper: given pretrained MLP blocks {MLP_pretrained_i}, initialise the CMS levels with them and
update level i with Eq. 71 (once every C^(i) steps, summed error terms); as the inner learning
rates go to 0 the pretrained model is recovered.

Our reading (D-E01-1): the pretrained decoder's per-layer MLPs ARE the CMS blocks. Layer j's MLP
is assigned to level (j mod K) ("interleave") so each frequency spans the whole depth. This keeps
the function exactly equal to the pretrained model at initialisation.
Inner optimiser (D-E01-2): Eq. 71 allows "an arbitrary optimizer"; we use one AdamW per level
(lr fixed a priori, no weight decay) on the NTP loss of the stream, the standard choice for
fine-tuning LLM weights. Attention, embeddings and norms stay frozen.
"""
from __future__ import annotations

import torch


def decoder_layers(model):
    for path in ("model.layers", "transformer.h", "gpt_neox.layers"):
        obj = model
        try:
            for p in path.split("."):
                obj = getattr(obj, p)
            return list(obj)
        except AttributeError:
            continue
    raise ValueError("unsupported architecture: cannot find decoder layers")


class ContinuumMemoryAdapter:
    def __init__(self, model, level_chunks: list[int], assignment: str = "interleave",
                 lr: float = 2e-5, betas=(0.9, 0.999)):
        self.chunks = list(level_chunks)
        self.base = min(self.chunks)
        assert all(c % self.base == 0 for c in self.chunks), "chunks must be multiples of the base"
        layers = decoder_layers(model)
        K, n = len(self.chunks), len(layers)
        for p in model.parameters():
            p.requires_grad_(False)
        self.level_of_layer, level_params = [], [[] for _ in range(K)]
        for j, layer in enumerate(layers):
            lvl = j % K if assignment == "interleave" else min(K - 1, j * K // n)
            self.level_of_layer.append(lvl)
            for p in layer.mlp.parameters():
                if p.dtype != torch.float32:
                    raise TypeError("adaptable MLP weights must be fp32 (bf16 loses 2e-5 updates)")
                p.requires_grad_(True)
                level_params[lvl].append(p)
        self.params = level_params
        self.opts = [torch.optim.AdamW(ps, lr=lr, betas=betas, weight_decay=0.0)
                     for ps in level_params]
        self.tokens_seen = 0
        self.updates = [0] * K

    def n_adaptable(self) -> list[int]:
        return [sum(p.numel() for p in ps) for ps in self.params]

    def after_backward(self, n_tokens: int) -> None:
        """Call once per base chunk, after loss.backward(). Level l steps every C_l tokens with
        the gradients accumulated since its previous step (Eq. 71)."""
        assert n_tokens == self.base, "stream must be fed in base-chunk units"
        self.tokens_seen += n_tokens
        for lvl, (c, opt) in enumerate(zip(self.chunks, self.opts)):
            if self.tokens_seen % c == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)
                self.updates[lvl] += 1
