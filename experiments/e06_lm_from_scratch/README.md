# E06 — Table 2 at small scale: Hope vs Transformer++ from scratch (ready)

prepare_data -> profile_throughput -> train (x2 arms) -> evaluate (x2) -> summarize; `make e06`
runs them in order and is resumable. Configs: `configs/common.yaml` (shared, fixed a priori),
`transformer_110m.yaml`, `hope_110m.yaml`, and `*_smoke.yaml` (pipeline checks only).
Protocol: docs/EXPERIMENT_PLAN.md (E06); budget and cost guard: docs/COST_AND_FEASIBILITY.md.
