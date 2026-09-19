"""Collect E01 results into a markdown table and a Figure-6-style plot (accuracy vs. task)."""
import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/e01")
    a = ap.parse_args()
    lines = ["| dataset | arm | final avg acc | mean forgetting | level updates | minutes |",
             "|---|---|---:|---:|---|---:|"]
    curves = {}
    for ds_dir in sorted(glob.glob(os.path.join(a.results, "*"))):
        ds = os.path.basename(ds_dir)
        for f in sorted(glob.glob(os.path.join(ds_dir, "*.json"))):
            if f.endswith("setup.json"):
                continue
            r = json.load(open(f))
            name = r["arm"]["name"]
            fg = r["mean_forgetting"]
            lines.append(f"| {ds} | {name} | {r['final_avg_acc']:.3f} | "
                         f"{'-' if fg is None else f'{fg:.3f}'} | {r['level_updates']} | "
                         f"{r['wall_seconds'] / 60:.1f} |")
            curves.setdefault(ds, {})[name] = [p["avg_acc"] for p in r["results_by_eval_point"]]
    table = "\n".join(lines)
    print(table)
    with open(os.path.join(a.results, "summary.md"), "w") as f:
        f.write(table + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, max(1, len(curves)), figsize=(4.5 * max(1, len(curves)), 3.6))
    axes = axes if len(curves) > 1 else [axes]
    for ax, (ds, arms) in zip(axes, curves.items()):
        for name, ys in arms.items():
            ax.plot(range(1, len(ys) + 1), ys, marker="o", label=name)
        ax.set_title(ds); ax.set_xlabel("tasks seen"); ax.set_ylabel("acc. on seen classes")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(a.results, "figure6_reproduction.png"), dpi=140)


if __name__ == "__main__":
    main()
