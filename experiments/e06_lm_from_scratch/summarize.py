"""Side-by-side E06 table (Transformer++ vs Hope): params, tokens, cost, val ppl, zero-shot."""
import json
import os

RUNS = {"Transformer++": "/workspace/runs/e06_transformer_110m",
        "Hope": "/workspace/runs/e06_hope_110m"}
METRIC = {"wikitext": "word_perplexity,none", "lambada_openai": "acc,none"}


def pick(task, res):
    if "error" in res:
        return "err"
    key = METRIC.get(task, "acc_norm,none" if "acc_norm,none" in res else "acc,none")
    v = res.get(key)
    return "-" if v is None else f"{v:.3f}" if v < 10 else f"{v:.1f}"


def main():
    rows, tasks = {}, None
    for name, d in RUNS.items():
        s = json.load(open(os.path.join(d, "train_summary.json"))) if os.path.exists(
            os.path.join(d, "train_summary.json")) else {}
        e = json.load(open(os.path.join(d, "eval.json"))) if os.path.exists(
            os.path.join(d, "eval.json")) else {}
        tasks = tasks or list(e.get("zero_shot", {}).keys())
        rows[name] = (s, e)
    head = ["arm", "params", "tokens", "GPU h", "status", "val ppl"] + (tasks or [])
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, (s, e) in rows.items():
        zs = e.get("zero_shot", {})
        out.append("| " + " | ".join([
            name, f"{s.get('params', {}).get('total', 0) / 1e6:.1f}M", f"{s.get('tokens', 0) / 1e9:.2f}B",
            f"{s.get('wall_hours', 0):.1f}", s.get("status", "-"),
            f"{e['val_ppl']:.2f}" if "val_ppl" in e else "-"] +
            [pick(t, zs.get(t, {"error": 1})) for t in tasks or []]) + " |")
    txt = "\n".join(out)
    print(txt)
    os.makedirs("results", exist_ok=True)
    open("results/e06_summary.md", "w").write(txt + "\n")


if __name__ == "__main__":
    main()
