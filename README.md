# ChIEDiff: chemistry-gated interaction-energy diffusion for structure-based 3D molecular generation

A pocket-conditioned 3D diffusion model that adds a chemistry-gated, Vina-derived
intermolecular energy to the training objective and steers sampling with a learned
affinity-head gradient.

## Installation

```bash
conda env create -f envs/environment-train.yml
conda activate ged-train
pip install -e .
```

Re-measuring needs three more environments and two external binaries, SMINA and GNINA 1.1.
See [`envs/README.md`](envs/README.md).

```bash
conda env create -f envs/environment-vina.yml
conda env create -f envs/environment-posecheck.yml
conda env create -f envs/environment-posebusters.yml
```

## Data and weights

[10.5281/zenodo.23029438](https://doi.org/10.5281/zenodo.23029438) holds the trained weights,
the generated molecules, our measurement outputs, the test receptors and the processed
training set.

```bash
python tools/download_archive.py --doi 10.5281/zenodo.23029438
gunzip data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb.gz
```

See [`data/README.md`](data/README.md).

## Reproducing the paper's tables

```bash
jupyter notebook reproduction.ipynb
```

Section 1 prints every table from the summary CSVs in `results/`. Section 2 recomputes them
from the generated molecules.

## Training

```bash
python bin/train.py --config configs/train.yml --tag myrun
python bin/train.py --config configs/train.yml --tag myrun --resume logs_diffusion/train_myrun
```

The released checkpoint is iteration 1,195,000, batch size 2 with 2 accumulation steps, about
a week on one 24 GB card.

## Generating molecules

```bash
bash bin/sample_local.sh --ckpt archive/chiediff_weights_iter1195000.pt --pockets 0-4
python scripts/group_samples.py --base results/generated --out results/myrun.pt
```

About an hour per pocket at 100 molecules on a 24 GB card. Only `head1_only` guidance is
implemented, with positions on grad log A-hat and atom types on grad A-hat.

## Evaluating generated molecules

```bash
bash bin/measure.sh --pt results/myrun.pt --tag myrun --pockets 0-4
```

Runs Vina, SMINA, Vinardo, GNINA, pose RMSD, PoseCheck, ProLIF, PoseBusters, PLIP, MMFF strain
and the chemical properties. `--pockets` defaults to 0-4; pass `0-99` for the full test set.
See [`docs/reproduce.md`](docs/reproduce.md).

## Requirements

Linux, Python 3.9, one NVIDIA GPU. PyTorch 1.11.0 + CUDA 11.3, PyTorch Geometric 2.0.4,
torch-scatter 2.0.9, RDKit 2024.03.6, NumPy 1.23.1, Open Babel 3.1.1. Evaluation adds
AutoDock Vina 1.2.2, Meeko 0.6.1, SMINA, GNINA 1.1, PLIP 3.0.0, PoseCheck 1.3.1,
ProLIF 2.2.0, PoseBusters 0.6.5.

Four things that change the numbers if you get them wrong:

- Set `OMP_NUM_THREADS=1`. The sampler does not reproduce individual molecules in any case.
- ProLIF must be 2.2.0. Version 2.1.0 changed the hydrophobic SMARTS: 2.04 contacts per
  molecule against 1.44.
- Measure against `data/test_set_chainfix`. Twelve receptors encode their chain copies only
  in the segID column, which Biopython ignores.
- Cohorts are not interchangeable: 9,999 sampled, 9,857 exported, 8,340 docking-successful,
  8,069 strict. Each table states its own n.

## Built on

[TargetDiff](https://github.com/guanjq/targetdiff) and
[KGDiff](https://github.com/CMACH508/KGDiff), both MIT, with coordinate-to-molecule
reconstruction from [liGAN](https://github.com/mattragoza/liGAN), GPL-2.0. Per-file
provenance and the compared models' licences are in [`NOTICE.md`](NOTICE.md).

## License

MIT, except `gated_energy_diffusion/utils/reconstruct.py`, which is GPL-2.0.

## Citing

See [`CITATION.cff`](CITATION.cff).

## Contact

Hyoungjoon Park and Sanghyun Park (sanghyun@yonsei.ac.kr), Department of Computer Science,
Yonsei University.
