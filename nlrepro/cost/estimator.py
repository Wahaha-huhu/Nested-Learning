"""Planning arithmetic (docs/COST_AND_FEASIBILITY.md). Estimates, not measurements.

    python -m nlrepro.cost.estimator            # prints the tables used in the docs
"""
from __future__ import annotations

PRICES = {"rtx4090": 0.74, "a100_80gb": 1.59}              # USD/h, Runpod list, 2026-09-19
EFFECTIVE_TFLOPS = {"rtx4090": (40, 100), "a100_80gb": (80, 160)}   # assumptions, unmeasured


def train_flops(n_params: float, tokens: float, n_layers: int = 0, d_model: int = 0,
                seq_len: int = 0) -> float:
    """6NT plus the causal-attention term 6 * n_layers * seq_len * d_model per token."""
    return 6 * n_params * tokens + 6 * n_layers * seq_len * d_model * tokens


def hours(flops: float, tflops: float, overhead: float = 1.0) -> float:
    return flops * overhead / (tflops * 1e12) / 3600


def cost_range(flops: float, gpu: str, overhead: float = 1.0) -> tuple:
    lo_t, hi_t = EFFECTIVE_TFLOPS[gpu]
    h_hi, h_lo = hours(flops, lo_t, overhead), hours(flops, hi_t, overhead)
    return h_lo, h_hi, h_lo * PRICES[gpu], h_hi * PRICES[gpu]


def state_gib(n_params: float, bytes_per_param: int = 16) -> float:
    return n_params * bytes_per_param / 2**30


def from_throughput(tokens: float, tokens_per_s: float, gpu: str, contingency: float = 0.25):
    h = tokens / tokens_per_s / 3600 * (1 + contingency)
    return h, h * PRICES[gpu]


def main() -> None:
    rows = [
        ("E06 Transformer++ 110M, 1B tok", 84.95e6 + 24.6e6, 1e9, 12, 768, 2048, [1.0]),
        ("E06 Hope 110M, 1B tok", 86.05e6 + 24.6e6, 1e9, 0, 0, 0, [2.0, 5.0]),
        ("Paper Table 2: 760M, 30B tok", 760e6, 30e9, 0, 0, 0, [1.0, 2.0, 5.0]),
        ("Paper Table 2: 1.3B, 100B tok", 1.3e9, 100e9, 0, 0, 0, [1.0, 2.0, 5.0]),
        ("Paper Fig. 6: 3B, 15B CPT tok", 3e9, 15e9, 0, 0, 0, [1.0]),
    ]
    print(f"{'workload':34s} {'ovh':>4s} {'4090 h':>14s} {'4090 $':>14s} {'A100 h':>14s} {'A100 $':>14s}")
    for name, n, t, nl, d, s, ovhs in rows:
        f = train_flops(n, t, nl, d, s)
        for o in ovhs:
            a = cost_range(f, "rtx4090", o)
            b = cost_range(f, "a100_80gb", o)
            print(f"{name:34s} {o:4.0f} {a[0]:6.1f}-{a[1]:<7.1f} {a[2]:6.0f}-{a[3]:<7.0f} "
                  f"{b[0]:6.1f}-{b[1]:<7.1f} {b[2]:6.0f}-{b[3]:<7.0f}")
    print("\nfull-training state floor (16 B/param):")
    for n in (0.11e9, 0.5e9, 1.3e9, 3e9, 8e9):
        print(f"  {n / 1e9:4.2f}B params -> {state_gib(n):6.1f} GiB")


if __name__ == "__main__":
    main()
