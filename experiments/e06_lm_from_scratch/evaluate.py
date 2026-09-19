"""Evaluate a finished E06 arm: held-out FineWeb-Edu ppl + zero-shot lm-eval tasks (Table 2).

    python -m experiments.e06_lm_from_scratch.evaluate --config .../configs/hope_110m.yaml
"""
import argparse
import math
import os

import torch

from nlrepro.eval.lm_harness import run_harness
from nlrepro.models import build_model
from nlrepro.train.lm_trainer import evaluate_val
from nlrepro.utils.common import environment, load_config, write_json

TASKS = ["wikitext", "lambada_openai", "piqa", "hellaswag", "winogrande", "arc_easy",
         "arc_challenge", "social_iqa", "boolq"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None, help="default: run_dir/final.pt")
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    ap.add_argument("--limit", type=int, default=None, help="per-task example cap (smoke)")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    path = a.ckpt or os.path.join(cfg["run_dir"], "final.pt")
    if not os.path.exists(path):
        summ = os.path.join(cfg["run_dir"], "train_summary.json")
        status = open(summ).read() if os.path.exists(summ) else "no train_summary.json"
        raise SystemExit(f"{path} not found: training did not complete.\n{status}\n"
                         f"see {cfg['run_dir']}/train_log.jsonl (pass --ckpt .../checkpoint.pt "
                         "to evaluate a partial run deliberately)")
    ck = torch.load(path, map_location=dev, weights_only=False)
    model = build_model(cfg["model"]).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    t = cfg["train"]
    vl = evaluate_val(model, os.path.join(cfg["data"]["dir"], "val.bin"), t["seq_len"],
                      1024, t["micro_batch"], dev, torch.bfloat16)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["data"]["tokenizer"])
    res = {"val_loss": vl, "val_ppl": math.exp(vl),
           "zero_shot": run_harness(model, tok, a.tasks, dev, batch_size=16, limit=a.limit),
           "env": environment()}
    write_json(os.path.join(cfg["run_dir"], "eval.json"), res)
    print(res)


if __name__ == "__main__":
    main()
