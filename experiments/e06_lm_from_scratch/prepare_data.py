"""Tokenise FineWeb-Edu once for both E06 arms (CPU-bound; ~1B tokens).

    python -m experiments.e06_lm_from_scratch.prepare_data --config .../configs/common.yaml
"""
import argparse
import os

from nlrepro.data.tokens import prepare
from nlrepro.utils.common import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    d = load_config(a.config, a.overrides)["data"]
    if os.path.exists(os.path.join(d["dir"], "meta.json")):
        print(f"{d['dir']} already prepared; delete it to rebuild")
        return
    meta = prepare(d["dir"], d["tokenizer"], d["dataset"], d["subset"], d["train_tokens"],
                   d["val_tokens"], workers=a.workers)
    print(meta)


if __name__ == "__main__":
    main()
