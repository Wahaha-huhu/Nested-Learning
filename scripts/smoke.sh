#!/usr/bin/env bash
# End-to-end pipeline check on the GPU (~15 min, < $0.25 on a 4090). Numbers are NOT results.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/workspace/hf_cache} HF_DATASETS_TRUST_REMOTE_CODE=1
C6=experiments/e06_lm_from_scratch/configs
SKIP_E01=${SKIP_E01:-0}
if [ "$SKIP_E01" = "1" ]; then echo "== E01 smoke skipped"; else
echo "== E01 smoke (includes the cached-vs-uncached scoring self-check)"
python -m experiments.e01_class_incremental.run \
  --config experiments/e01_class_incremental/configs/smoke.yaml --out results/smoke/e01
fi
echo "== E06 smoke data (5M tokens)"
python -m experiments.e06_lm_from_scratch.prepare_data --config $C6/transformer_smoke.yaml
for a in transformer hope; do
  echo "== E06 smoke train: $a"
  rm -rf /workspace/runs/e06_${a}_smoke
  python -m experiments.e06_lm_from_scratch.train --config $C6/${a}_smoke.yaml
  if [ ! -f /workspace/runs/e06_${a}_smoke/final.pt ]; then
    echo "!! $a smoke training did not complete:"
    cat /workspace/runs/e06_${a}_smoke/train_summary.json
    tail -n 5 /workspace/runs/e06_${a}_smoke/train_log.jsonl
    exit 1
  fi
  python -m experiments.e06_lm_from_scratch.evaluate --config $C6/${a}_smoke.yaml \
    --tasks lambada_openai piqa --limit 32
done
echo "== full-size throughput profile (decides whether E06 fits the budget)"
python -m experiments.e06_lm_from_scratch.profile_throughput --config $C6/transformer_110m.yaml
python -m experiments.e06_lm_from_scratch.profile_throughput --config $C6/hope_110m.yaml
echo "smoke ok"
