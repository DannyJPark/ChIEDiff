# `make` with no target prints this list.

.DEFAULT_GOAL := help
SHELL := /bin/bash

.PHONY: help install archive measure clean

help:  ## show this list
	@grep -hE '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[1m%-12s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "  Reproduce the tables   jupyter notebook reproduction.ipynb   (minutes, CPU)"
	@echo "  Train                  python bin/train.py --config configs/train.yml --tag myrun"
	@echo "  Sample                 bash bin/sample_local.sh --ckpt <weights> --pockets 0-4"
	@echo
	@echo "  Read docs/reproduce.md first. One GPU is enough and no cluster is needed."

install:  ## create the training environment and install the package into it
	conda env create -f envs/environment-train.yml
	@echo "Then: conda activate ged-train && pip install --no-build-isolation --no-deps plip==3.0.0 && pip install --no-deps -e ."
	@echo "The evaluation environments are separate; see envs/README.md."

archive:  ## download the weights, molecules, measurements and receptors
	python tools/download_archive.py --doi 10.5281/zenodo.23029438

measure:  ## measure the archived molecules on five pockets
	bash bin/measure.sh --pt archive/chiediff_generated_molecules.pt --tag chiediff --pockets 0-4

clean:  ## remove Python build products and notebook state
	rm -rf build dist *.egg-info .ipynb_checkpoints
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
