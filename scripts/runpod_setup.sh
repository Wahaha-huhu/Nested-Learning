#!/usr/bin/env bash
# One-time setup on a Runpod PyTorch pod. Repo is expected at /workspace/nested-learning-repro
# (network volume, so data/checkpoints survive pod restarts).
set -euo pipefail
cd "$(dirname "$0")/.."
python -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, torch.cuda.get_device_name(0))"
pip install -q -r requirements.txt
pip install -q -e .
mkdir -p /workspace/data /workspace/runs /workspace/hf_cache results
# Keep HF downloads on the persistent volume.
grep -q HF_HOME ~/.bashrc || echo 'export HF_HOME=/workspace/hf_cache' >> ~/.bashrc
grep -q HF_DATASETS_TRUST_REMOTE_CODE ~/.bashrc || echo 'export HF_DATASETS_TRUST_REMOTE_CODE=1' >> ~/.bashrc
export HF_HOME=/workspace/hf_cache HF_DATASETS_TRUST_REMOTE_CODE=1
python -m pytest -q
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "setup ok. next: make smoke"
