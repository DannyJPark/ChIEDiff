# Front door. `make` with no target prints this list.
#
# The tiers differ enormously in cost, so they are separate targets rather than one
# pipeline: `tables` is a minute on a laptop, `remeasure` is days across four conda
# environments, `train` is over a million iterations on a GPU.

.DEFAULT_GOAL := help
SHELL := /bin/bash

.PHONY: help tables tables-dry remeasure measure sample verify check check-release clean

help:  ## show this list
	@grep -hE '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[1m%-16s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "  Tier 1  make tables      reproduce every table from the shipped CSVs (~1 min, CPU)"
	@echo "  Tier 2  bin/measure.sh then bin/remeasure.sh    re-measure locally (hours-days)"
	@echo "  Tier 3  python bin/train.py --config configs/train.yml --tag myrun   (days, 1 GPU)"
	@echo
	@echo "  No cluster needed. slurm/ holds optional launchers; see docs/reproduce.md."

tables:  ## shipped CSVs -> all 22 generated tables
	bash bin/make_tables.sh

tables-dry:  ## render the tables without writing them into the LaTeX targets
	bash bin/make_tables.sh --dry

remeasure:  ## raw measurement outputs -> the 19 source CSVs -> tables
	bash bin/remeasure.sh

measure:  ## generated samples -> docking / PLIP / PoseCheck / PoseBusters (submits to Slurm)
	bash bin/measure.sh

verify:  ## equivalence controls against the original implementation (submits to Slurm)
	@echo "Submitting the equivalence controls. These MUST run on a compute node:"
	@echo "  A  loss and gradient equivalence, bitwise, over seven timesteps"
	@echo "  A' 50-iteration end-to-end training comparison"
	@echo "  B  sampling equivalence, bitwise, at OMP_NUM_THREADS=1"
	@echo "  C  the chemistry-gate assertion fires when the transform is absent"
	@echo "  D  per-term energy linearity"
	sbatch slurm/verify_equivalence.sbatch

check:  ## every gate, no regeneration
	bash bin/check.sh

check-release:  ## as check, but also fails on unassigned DOIs and placeholder URLs
	bash bin/check.sh --release

clean:  ## remove LaTeX build products from the table harness
	rm -f tables/*.aux tables/*.log tables/*.out tables/*.toc \
	      tables/*.fls tables/*.fdb_latexmk tables/*.synctex.gz
