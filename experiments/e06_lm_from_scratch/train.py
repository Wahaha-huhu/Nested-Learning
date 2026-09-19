"""Train one E06 arm. Re-running the same command resumes from run_dir/checkpoint.pt.

    python -m experiments.e06_lm_from_scratch.train --config .../configs/hope_110m.yaml
"""
import argparse

from nlrepro.train.lm_trainer import train
from nlrepro.utils.common import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*", help="dotted key=value, e.g. guard.max_projected_hours=40")
    a = ap.parse_args()
    print(train(load_config(a.config, a.overrides)))


if __name__ == "__main__":
    main()
