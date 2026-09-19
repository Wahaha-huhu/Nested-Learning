"""E01 / Figure 6: class-incremental learning, ICL vs Hope-style CMS on a pretrained decoder.

Stage A (implemented here, no continued pretraining): the model reads one long stream
  task_1 examples | task_2 examples | ... | task_T examples      (format: data/intent.py)
Arms (identical stream, identical attention window, identical evaluation):
  icl       : frozen weights; test-time context = last `ctx_tokens` tokens of the stream
  cms_1lvl  : every layer's MLP updated online every C_0 tokens (single frequency control)
  cms_3lvl  : MLPs split into 3 levels updated every C_0, C_1, C_2 tokens (the CMS arm)
After each task: accuracy on test queries of all classes seen so far, candidates = seen labels.
Paper deviations: docs/DEVIATIONS.md (D-E01-*). Stage B (continued pretraining) is gated on A.
"""
from __future__ import annotations

import argparse
import gc
import os
import time

import torch

from nlrepro.data.intent import format_example, load_cil
from nlrepro.eval.cil_scoring import LabelScorer
from nlrepro.models.hf_cms import ContinuumMemoryAdapter
from nlrepro.utils.common import environment, load_yaml, seed_everything, write_json


def build_stream(tok, tasks, base_chunk: int, newline_id: int):
    """Token stream; every task segment is padded to a multiple of base_chunk with newline
    tokens whose loss is masked, so evaluation points coincide with chunk boundaries and no
    next-task example leaks into the 'after task t' state."""
    ids, lossmask, ends, raw_ends = [], [], [], []
    for t in tasks:
        for ex in t["train"]:
            x = tok(format_example(ex.text, ex.label), add_special_tokens=False)["input_ids"]
            ids += x
            lossmask += [1] * len(x)
        raw_ends.append(len(ids))
        pad = (-len(ids)) % base_chunk
        ids += [newline_id] * pad
        lossmask += [0] * pad
        ends.append(len(ids))
    return ids, lossmask, ends, raw_ends


def evaluate(scorer, tok, tasks, upto: int, ctx_ids: list[int], do_check: bool):
    seen = [c for t in tasks[:upto + 1] for c in t["classes"]]
    labels = sorted({ex.label for t in tasks[:upto + 1] for ex in t["test"]})
    label_idx = {l: i for i, l in enumerate(labels)}
    ctx = scorer.encode_context(ctx_ids)
    check_err = None
    per_task = []
    for ti in range(upto + 1):
        correct = 0
        for n, ex in enumerate(tasks[ti]["test"]):
            q = tok(format_example(ex.text), add_special_tokens=False)["input_ids"]
            if do_check and check_err is None:
                check_err = scorer.self_check(ctx_ids, q, labels)
            s = scorer.score(ctx, q, labels)
            correct += int(int(s.argmax()) == label_idx[ex.label])
        per_task.append(correct / len(tasks[ti]["test"]))
    n_q = [len(t["test"]) for t in tasks[:upto + 1]]
    avg = sum(a * n for a, n in zip(per_task, n_q)) / sum(n_q)
    return {"per_task_acc": per_task, "avg_acc": avg, "n_classes": len(seen),
            "cache_check_err": check_err}


def run_arm(cfg, arm, tasks, stream, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    seed_everything(cfg["seed"])
    tok = AutoTokenizer.from_pretrained(cfg["model_id"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"], torch_dtype=torch.float32, attn_implementation="sdpa").to(device)
    model.eval()          # no dropout; gradients still flow for the adaptable MLPs
    adapter = None
    if arm.get("level_chunks"):
        adapter = ContinuumMemoryAdapter(model, arm["level_chunks"], arm.get("assignment",
                                         "interleave"), lr=cfg["inner_lr"])
    scorer = LabelScorer(model, tok, device)
    ids, lossmask, ends, raw_ends = stream
    base, W = cfg["base_chunk"], cfg["window_tokens"]
    ids_t = torch.tensor(ids)
    mask_t = torch.tensor(lossmask, dtype=torch.bool)

    matrix, t0, pos, stream_losses = [], time.time(), 0, []
    for ti, end in enumerate(ends):
        while pos < end:
            chunk_end = pos + base
            if adapter is not None:
                start = max(0, chunk_end - W)
                x = ids_t[start:chunk_end][None].to(device)
                y = x.clone()
                y[0, :pos - start] = -100                           # context tokens: no loss
                y[0, pos - start:][~mask_t[pos:chunk_end].to(device)] = -100
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    loss = model(input_ids=x, labels=y).loss
                loss.backward()
                adapter.after_backward(base)
                stream_losses.append(float(loss))
            pos = chunk_end
        ctx = ids[max(0, raw_ends[ti] - cfg["ctx_tokens"]):raw_ends[ti]]
        res = evaluate(scorer, tok, tasks, ti, ctx, do_check=(ti == 0))
        res["after_task"] = ti
        matrix.append(res)
        print(f"[{arm['name']}] after task {ti + 1}/{len(tasks)}: avg_acc={res['avg_acc']:.3f}",
              flush=True)

    T = len(tasks)
    final = matrix[-1]["per_task_acc"]
    forgetting = [max(matrix[t]["per_task_acc"][i] for t in range(i, T - 1)) - final[i]
                  for i in range(T - 1)] if T > 1 else []
    out = {"arm": arm, "results_by_eval_point": matrix,
           "final_avg_acc": matrix[-1]["avg_acc"],
           "mean_forgetting": sum(forgetting) / len(forgetting) if forgetting else None,
           "stream_tokens": len(ids), "stream_loss_curve": stream_losses,
           "level_updates": adapter.updates if adapter else None,
           "adaptable_params_per_level": adapter.n_adaptable() if adapter else None,
           "wall_seconds": time.time() - t0}
    del model, adapter, scorer
    gc.collect()
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="results/e01")
    ap.add_argument("--arms", nargs="*", help="subset of arm names")
    ap.add_argument("overrides", nargs="*", help="dotted key=value overrides")
    args = ap.parse_args()
    cfg = load_yaml(args.config, args.overrides)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model_id"])
    tasks, names = load_cil(cfg["dataset"], cfg["n_tasks"], cfg["train_per_class"],
                            cfg["test_per_class"], seed=cfg["seed"])
    nl = tok("\n", add_special_tokens=False)["input_ids"][-1]
    stream = build_stream(tok, tasks, cfg["base_chunk"], nl)
    out_dir = os.path.join(args.out, cfg["dataset"])
    os.makedirs(out_dir, exist_ok=True)
    write_json(os.path.join(out_dir, "setup.json"), {
        "config": cfg, "env": environment(), "stream_tokens": len(stream[0]),
        "task_classes": [[names[c] for c in t["classes"]] for t in tasks]})
    for arm in cfg["arms"]:
        if args.arms and arm["name"] not in args.arms:
            continue
        res = run_arm(cfg, arm, tasks, stream, device)
        write_json(os.path.join(out_dir, f"{arm['name']}.json"), res)
        print(f"== {arm['name']}: final avg acc {res['final_avg_acc']:.3f}, "
              f"forgetting {res['mean_forgetting']}, {res['wall_seconds'] / 60:.1f} min")


if __name__ == "__main__":
    main()
