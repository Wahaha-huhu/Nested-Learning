"""Pretraining tokens for E06.

Paper (Sec. 9.3): FineWeb-Edu + long-context documents, 32K vocabulary. We use the FineWeb-Edu
`sample-10BT` subset only (D-E06-2) and the Llama-2 32K SentencePiece tokenizer as shipped with
TinyLlama (Apache-2.0, not gated). Documents are separated by EOS; stored as uint16.
"""
from __future__ import annotations

import json
import os
from multiprocessing import Pool

import numpy as np

_TOK = None


def _init(tok_id: str) -> None:
    global _TOK
    from transformers import AutoTokenizer
    _TOK = AutoTokenizer.from_pretrained(tok_id)


def _encode(texts: list[str]) -> list[np.ndarray]:
    ids = _TOK(texts, add_special_tokens=False)["input_ids"]
    eos = _TOK.eos_token_id
    return [np.array(x + [eos], dtype=np.uint16) for x in ids]


def _batches(ds, bs: int):
    buf = []
    for ex in ds:
        buf.append(ex["text"])
        if len(buf) == bs:
            yield buf
            buf = []
    if buf:
        yield buf


def prepare(out_dir: str, tokenizer_id: str, dataset: str, subset: str, train_tokens: int,
            val_tokens: int, workers: int = 8, docs_per_batch: int = 512) -> dict:
    """Stream the dataset once: first `val_tokens` go to val.bin, the next `train_tokens` to
    train.bin. Resumable only from scratch (delete partial files)."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    os.makedirs(out_dir, exist_ok=True)
    vocab = AutoTokenizer.from_pretrained(tokenizer_id).vocab_size
    assert vocab < 2 ** 16, "uint16 storage requires vocab < 65536"
    ds = load_dataset(dataset, name=subset, split="train", streaming=True)
    targets = {"val": val_tokens, "train": train_tokens}
    written = {"val": 0, "train": 0}
    files = {k: open(os.path.join(out_dir, f"{k}.bin"), "wb") for k in targets}
    split, n_docs = "val", 0
    with Pool(workers, initializer=_init, initargs=(tokenizer_id,)) as pool:
        for arrs in pool.imap(_encode, _batches(ds, docs_per_batch), chunksize=1):
            for a in arrs:
                if split == "val" and written["val"] >= targets["val"]:
                    split = "train"
                if split == "train" and written["train"] >= targets["train"]:
                    break
                files[split].write(a.tobytes())
                written[split] += len(a)
                n_docs += 1
            if written["train"] >= targets["train"]:
                break
            if n_docs % (docs_per_batch * 200) < docs_per_batch:
                print(f"docs={n_docs} val={written['val']:,} train={written['train']:,}",
                      flush=True)
    for f in files.values():
        f.close()
    meta = {"tokenizer": tokenizer_id, "vocab_size": vocab, "dataset": dataset,
            "subset": subset, "tokens": written, "documents": n_docs, "dtype": "uint16"}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


class TokenSampler:
    """Uniformly random (seeded) windows of seq_len+1 tokens from a uint16 memmap.
    Both E06 arms use the same seed, hence the identical sequence of training batches."""

    def __init__(self, path: str, seq_len: int, batch_size: int, seed: int, device: str):
        import torch
        self.torch = torch
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq_len, self.bs, self.device = seq_len, batch_size, device
        self.gen = torch.Generator().manual_seed(seed)
        self.high = len(self.data) - seq_len - 1
        assert self.high > 0, f"{path} too small"

    def state_dict(self) -> dict:
        return {"gen": self.gen.get_state()}

    def load_state_dict(self, s: dict) -> None:
        self.gen.set_state(s["gen"])

    def next(self):
        torch = self.torch
        offs = torch.randint(0, self.high, (self.bs,), generator=self.gen).tolist()
        buf = np.stack([self.data[o:o + self.seq_len + 1].astype(np.int64) for o in offs])
        t = torch.from_numpy(buf)
        if str(self.device).startswith("cuda"):
            t = t.pin_memory().to(self.device, non_blocking=True)
        return t[:, :-1], t[:, 1:]


def fixed_val_batches(path: str, seq_len: int, n_seqs: int, batch_size: int, device: str):
    """Deterministic, non-overlapping validation windows from the start of val.bin."""
    import torch
    data = np.memmap(path, dtype=np.uint16, mode="r")
    n = min(n_seqs, (len(data) - 1) // seq_len)
    for i in range(0, n, batch_size):
        rows = [data[j * seq_len: j * seq_len + seq_len + 1].astype(np.int64)
                for j in range(i, min(i + batch_size, n))]
        t = torch.from_numpy(np.stack(rows)).to(device)
        yield t[:, :-1], t[:, 1:]
