# Running on Runpod

1. **Storage first.** Create a network volume (20-50 GB is enough for this round) in the same
   region as the GPU and mount it at `/workspace`. Data, checkpoints and the HF cache live there,
   so a stopped or preempted pod loses nothing. Stopped pods and volumes still bill for storage.
2. **Pod.** RTX 4090 (24 GB), a Runpod PyTorch template (torch >= 2.3, CUDA 12.x), >= 8 vCPUs
   (tokenisation is CPU-bound), container disk 20 GB.
3. **Setup.**
   ```bash
   cd /workspace && git clone <your-remote>/nested-learning-repro && cd nested-learning-repro
   bash scripts/runpod_setup.sh          # installs pinned deps, runs unit tests incl. torch tests
   ```
   Llama checkpoints are gated; the default configs need no HF token.
4. **Smoke test (always first, ~15 min):** `make smoke`. It must end with `smoke ok`, pass the
   cached-scoring self-check, and print the full-size throughput profile of both E06 arms.
   Decide whether the projected Hope cost is acceptable before step 6.
5. **E01:** `make e01` (~1 h). Results: `results/e01/summary.md`, `results/e01/figure6_reproduction.png`.
6. **E06:** start `tmux`, then `make e06`. Tokenisation runs first, then the Transformer++ arm,
   then Hope, each followed by evaluation. If the pod stops, re-run `make e06`: finished steps are
   skipped and training resumes from `checkpoint.pt` (saved every 250 steps, ~1 h of Hope time at
   most). Monitor with `tail -f /workspace/runs/e06_*/train_log.jsonl`; the `projection` event after
   30 steps shows projected hours and USD.
7. **Bring back:** `results/`, `/workspace/runs/*/{train_log.jsonl,train_summary.json,eval.json,run_info.json}`.
   Checkpoints (~0.5-1.3 GB each) only if needed for E04/E05/E07.
