# Common entry points. On Runpod: `bash scripts/runpod_setup.sh` first.
PY ?= python
CFG6 = experiments/e06_lm_from_scratch/configs

test:            ## unit tests (torch tests are skipped where torch is missing)
	$(PY) -m pytest -q

e00:             ## CPU, seconds
	$(PY) -m experiments.e00_delta_momentum.run --out results/e00

smoke:           ## ~15 min end-to-end check of E01 + E06 pipelines on the GPU (not results)
	bash scripts/smoke.sh

e01:             ## Stage A on CLINC, Banking, DBpedia (~1 GPU-hour on a 4090)
	bash scripts/run_e01.sh

e06-data:        ## tokenise 1B FineWeb-Edu tokens (CPU-bound)
	$(PY) -m experiments.e06_lm_from_scratch.prepare_data --config $(CFG6)/common.yaml

e06-profile:     ## measured throughput -> projected cost, both arms
	$(PY) -m experiments.e06_lm_from_scratch.profile_throughput --config $(CFG6)/transformer_110m.yaml
	$(PY) -m experiments.e06_lm_from_scratch.profile_throughput --config $(CFG6)/hope_110m.yaml

e06:             ## train + evaluate both arms (resumable)
	bash scripts/run_e06.sh

cost:            ## planning tables
	$(PY) -m nlrepro.cost.estimator
