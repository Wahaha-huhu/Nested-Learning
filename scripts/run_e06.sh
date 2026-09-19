#!/usr/bin/env bash
# E06: both arms, sequentially. Resumable: re-run after an interruption.
# Run inside tmux; the cost guard aborts an arm whose projected time exceeds its cap.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results
export HF_HOME=${HF_HOME:-/workspace/hf_cache} HF_DATASETS_TRUST_REMOTE_CODE=1
C6=experiments/e06_lm_from_scratch/configs
python -m experiments.e06_lm_from_scratch.prepare_data --config $C6/common.yaml
for a in transformer hope; do
  python -m experiments.e06_lm_from_scratch.train --config $C6/${a}_110m.yaml 2>&1 | tee -a results/e06_${a}_train.log
  if [ -f /workspace/runs/e06_${a}_110m/final.pt ]; then
    python -m experiments.e06_lm_from_scratch.evaluate --config $C6/${a}_110m.yaml 2>&1 | tee -a results/e06_${a}_eval.log
  fi
done
python -m experiments.e06_lm_from_scratch.summarize
