"""Class-incremental text classification data for E01 (paper Fig. 6).

Paper datasets: CLINC150, Banking77, DBpedia (70-class level-2, 10K train / 1K test subsample),
following the protocol of Momeni et al. (2025, InCA). The exact class orders/splits of the paper
are not published, so we fix our own with a declared seed (D-E01-4).
"""
from __future__ import annotations

import random
from dataclasses import dataclass

DATASETS = {
    # key: (hf id, config, text field, label field, test split)
    "clinc": ("clinc/clinc_oos", "plus", "text", "intent", "test"),
    "banking": ("PolyAI/banking77", None, "text", "label", "test"),
    "dbpedia": ("DeveloperOats/DBPedia_Classes", None, "text", "l2", "test"),
}


@dataclass
class Example:
    text: str
    label: str
    cls: int


def _label_names(ds, field):
    feat = ds.features[field]
    if hasattr(feat, "names"):
        return list(feat.names), True
    return sorted(set(ds[field])), False


def load_cil(name: str, n_tasks: int, train_per_class: int, test_per_class: int,
             seed: int = 0, max_query_chars: int = 400):
    """Returns (tasks, class_names) where tasks[t] = {"classes", "train", "test"}."""
    from datasets import load_dataset
    hf_id, config, tf, lf, test_split = DATASETS[name]
    kw = {"trust_remote_code": True}
    train = load_dataset(hf_id, config, split="train", **kw)
    test = load_dataset(hf_id, config, split=test_split, **kw)
    names, is_classlabel = _label_names(train, lf)

    def rows(ds):
        by = {}
        for ex in ds:
            y = ex[lf]
            key = y if is_classlabel else names.index(y)
            by.setdefault(key, []).append(ex[tf].strip()[:max_query_chars])
        return by

    tr, te = rows(train), rows(test)
    classes = [c for c in range(len(names)) if names[c] != "oos"]   # CLINC out-of-scope removed
    rng = random.Random(seed)
    rng.shuffle(classes)
    classes = [c for c in classes if len(tr.get(c, [])) >= train_per_class
               and len(te.get(c, [])) >= test_per_class]
    per_task = len(classes) // n_tasks
    classes = classes[:per_task * n_tasks]
    pretty = {c: names[c].replace("_", " ").strip() for c in classes}

    tasks = []
    for t in range(n_tasks):
        cs = classes[t * per_task:(t + 1) * per_task]
        trn, tst = [], []
        for c in cs:
            a, b = tr[c][:], te[c][:]
            rng.shuffle(a); rng.shuffle(b)
            trn += [Example(x, pretty[c], c) for x in a[:train_per_class]]
            tst += [Example(x, pretty[c], c) for x in b[:test_per_class]]
        rng.shuffle(trn)   # classes of a task are interleaved in the stream
        tasks.append({"classes": cs, "train": trn, "test": tst})
    return tasks, pretty


def format_example(text: str, label: str | None = None) -> str:
    """Stream format. The query part is shared by training examples and test prompts."""
    q = f"Query: {text}\nIntent:"
    return q if label is None else f"{q} {label}\n\n"
