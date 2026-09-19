#!/usr/bin/env bash
# E01 stage A, datasets in the paper's order of discussion, CLINC first. ~1 GPU-hour total.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results
export HF_HOME=${HF_HOME:-/workspace/hf_cache} HF_DATASETS_TRUST_REMOTE_CODE=1
for ds in clinc banking dbpedia; do
  python -m experiments.e01_class_incremental.run \
    --config experiments/e01_class_incremental/configs/${ds}.yaml --out results/e01 \
    2>&1 | tee -a results/e01_${ds}.log
done
python -m experiments.e01_class_incremental.summarize --results results/e01
