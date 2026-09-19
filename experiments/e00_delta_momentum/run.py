"""E00 / Figure 4 reproduction: standard vs delta momentum on psi(r, theta).

CPU only, float64, deterministic (no randomness at all). Usage:
    python -m experiments.e00_delta_momentum.run --out results/e00

The primary configuration is fixed a priori in config.yaml. The robustness grid is NOT a tuning
step: nothing is selected from it. It reports how often each delta convention beats standard
momentum over a declared grid of landscapes and optimiser settings.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from dataclasses import replace

import numpy as np
import yaml

from nlrepro.optim.toy_momentum import Landscape, OptConfig, describe, numeric_grad, run

HERE = os.path.dirname(os.path.abspath(__file__))


def outcome(std: dict, dlt: dict) -> str:
    s, d = std["converged_at"], dlt["converged_at"]
    if s is None and d is None:
        return "neither"
    if s is None:
        return "delta_only"
    if d is None:
        return "standard_only"
    if d < s:
        return "delta_faster"
    if d > s:
        return "standard_faster"
    return "tie"


def primary(cfg_yaml: dict, out: str) -> dict:
    land = Landscape(**cfg_yaml["landscape"])
    base = OptConfig(**cfg_yaml["optimizer"])
    p0 = np.array(base.start)
    g_err = float(np.max(np.abs(land.grad(p0) - numeric_grad(land, p0))))
    assert g_err < 1e-4, f"analytic gradient mismatch {g_err}"

    res = {"gradient_check_max_abs_err": g_err, "config": describe(land, base), "runs": {}}
    runs = {"standard": run(land, base, delta=False)}
    for conv in cfg_yaml["conventions"]:
        runs[f"delta_{conv}"] = run(land, replace(base, convention=conv), delta=True)
    for name, r in runs.items():
        res["runs"][name] = {
            "converged_at": r["converged_at"],
            "final_value": r["final_value"],
            "diverged": bool(r["diverged"]),
            "steps_run": len(r["values"]) - 1,
        }
        np.savez(os.path.join(out, f"trajectory_{name}.npz"),
                 trajectory=r["trajectory"], values=r["values"])
    plot(land, runs, os.path.join(out, "figure4_reproduction.png"))
    return res


def grid(cfg_yaml: dict) -> dict:
    g = cfg_yaml["robustness_grid"]
    base = OptConfig(**cfg_yaml["optimizer"])
    table, summary = [], {}
    for conv in cfg_yaml["conventions"]:
        counts: dict[str, int] = {}
        ratios = []
        betas = g["beta"] if conv == "normalized" else [None]
        for k, a, w, eta, alpha, beta in itertools.product(
                g["k"], g["a"], g["w"], g["eta"], g["alpha"], betas):
            land = Landscape(k=k, a=a, w=w)
            oc = replace(base, eta=eta, alpha=alpha, convention=conv,
                         **({"beta": beta} if beta is not None else {}))
            s, d = run(land, oc, False), run(land, oc, True)
            o = outcome(s, d)
            counts[o] = counts.get(o, 0) + 1
            if s["converged_at"] and d["converged_at"]:
                ratios.append(s["converged_at"] / d["converged_at"])
            table.append({"convention": conv, "k": k, "a": a, "w": w, "eta": eta,
                          "alpha": alpha, "beta": beta, "standard_steps": s["converged_at"],
                          "delta_steps": d["converged_at"], "outcome": o})
        summary[conv] = {
            "n_settings": sum(counts.values()),
            "outcomes": counts,
            "median_step_ratio_standard_over_delta": float(np.median(ratios)) if ratios else None,
            "n_both_converged": len(ratios),
        }
    return {"summary": summary, "table": table}


def plot(land: Landscape, runs: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    rr, tt = np.meshgrid(np.linspace(-4, 3.5, 300), np.linspace(-3, 3, 300))
    zz = rr ** 2 + land.k * (rr - tt + land.a * np.sin(land.w * rr)) ** 2
    ax1.contour(rr, tt, np.log10(zz + 1e-3), levels=30, cmap="Greys", linewidths=0.6)
    styles = {"standard": ("tab:blue", "-"), "delta_normalized": ("tab:red", "-"),
              "delta_gd_l2": ("tab:green", "--")}
    for name, r in runs.items():
        c, ls = styles.get(name, ("k", ":"))
        tr = r["trajectory"]
        ax1.plot(tr[:, 0], tr[:, 1], color=c, ls=ls, lw=1.0, label=name)
        ax2.semilogy(np.maximum(r["values"], 1e-12), color=c, ls=ls, lw=1.2,
                     label=f"{name} (conv. step {r['converged_at']})")
    ax1.plot(*runs["standard"]["trajectory"][0], "ko", ms=5)
    ax1.plot(0, 0, "k*", ms=10)
    ax1.set_xlabel("r"); ax1.set_ylabel("theta"); ax1.set_title("Trajectories on psi(r, theta)")
    ax1.legend(fontsize=8)
    ax2.set_xlabel("step"); ax2.set_ylabel("psi"); ax2.set_title("Objective")
    ax2.legend(fontsize=8)
    fig.suptitle("E00 (paper Fig. 4) - reconstructed constants, see docs/DEVIATIONS.md", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--out", default="results/e00")
    ap.add_argument("--skip-grid", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    res = primary(cfg, args.out)
    if not args.skip_grid:
        g = grid(cfg)
        res["robustness_summary"] = g["summary"]
        with open(os.path.join(args.out, "robustness_grid.json"), "w") as f:
            json.dump(g["table"], f, indent=1)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
