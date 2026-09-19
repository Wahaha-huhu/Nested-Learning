#!/usr/bin/env bash
# E06: both arms, sequentially. Resumable: re-run after an interruption.
# Run inside tmux; the cost guard aborts an arm whose projected time exceeds its cap.
# The micro-batch is chosen by the profiler (tokens per optimiser step stay fixed at 262,144),
# and the choice is saved in the run dir so a resumed run uses the same setting.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results
export HF_HOME=${HF_HOME:-/workspace/hf_cache} HF_DATASETS_TRUST_REMOTE_CODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C6=experiments/e06_lm_from_scratch/configs
python -m experiments.e06_lm_from_scratch.prepare_data --config $C6/common.yaml
for a in transformer hope; do
  RUN=/workspace/runs/e06_${a}_110m
  mkdir -p $RUN
  if [ ! -f $RUN/overrides.txt ]; then
    python -m experiments.e06_lm_from_scratch.profile_throughput --config $C6/${a}_110m.yaml \
      --write-overrides $RUN/overrides.txt 2>&1 | tee -a results/e06_${a}_profile.log
  fi
  echo "== $a overrides: $(cat $RUN/overrides.txt)"
  python -m experiments.e06_lm_from_scratch.train --config $C6/${a}_110m.yaml $(cat $RUN/overrides.txt) \
    2>&1 | tee -a results/e06_${a}_train.log
  if [ -f $RUN/final.pt ]; then
    python -m experiments.e06_lm_from_scratch.evaluate --config $C6/${a}_110m.yaml \
      train.micro_batch=8 2>&1 | tee -a results/e06_${a}_eval.log
  fi
done
python -m experiments.e06_lm_from_scratch.summarize
