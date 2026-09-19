# E01 — Figure 6, class-incremental learning (stage A, ready)

`python -m experiments.e01_class_incremental.run --config configs/clinc.yaml --out results/e01`
then `python -m experiments.e01_class_incremental.summarize --results results/e01`.
One JSON per arm (accuracy after every task, forgetting, level update counts, stream loss) plus a
Figure-6-style plot. Protocol and choices: docs/EXPERIMENT_PLAN.md (E01), docs/DEVIATIONS.md (D-E01-*).
