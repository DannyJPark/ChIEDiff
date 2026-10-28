# ChIEDiff — chemistry-gated interaction-energy diffusion for structure-based 3D molecular generation

A pocket-conditioned 3D diffusion model that adds a chemistry-gated, Vina-derived
intermolecular energy to the training objective, evaluated at the predicted clean
coordinates, and steers sampling with a learned affinity-head gradient (`head1_only`,
coordinate scale `s_x = 25`, atom-type scale `s_v = 100`).

The hydrophobic and donor-acceptor gates are computed from the ground-truth chemistry of the
training complex and held fixed, so the energy branch sends no gradient into the atom-type
head. This repository holds the model, the configuration the paper reports, the evaluation
pipeline, and a notebook that prints every table from the measured results.

The loss variants explored during development, the inactive second guidance head, and the
one-off diagnostics are not here. This is the single configuration the paper reports.

## Contents

```
gated_energy_diffusion/   the model
bin/                      train.py, sample.py, and the measurement driver
configs/train.yml         the published training configuration
scripts/                  the measurement workers
data/                     the split and the canonical pocket order; see data/README.md
results/                  the summary numbers behind the tables; see results/README.md
reproduction.ipynb        prints every table, from results/ and from the molecules
envs/                     four conda environments and why they cannot be merged
docs/reproduce.md         how to reproduce the tables, measure and train
tools/download_archive.py fetches and unpacks the data archive
slurm/                    optional batch launchers
Makefile                  install, archive, measure, clean. Run `make` for the list
```

`LICENSE`, `NOTICE.md`, `CITATION.cff` and `pyproject.toml` are at the top level.
`configs/` also holds `posebusters_dock_1thread.yml`, which the two PoseBusters workers read.

## Availability and requirements

| Field | Value |
|---|---|
| Project name | ChIEDiff |
| Project home page | https://github.com/DannyJPark/ChIEDiff |
| Archived version | see CITATION.cff |
| Data and weights | [10.5281/zenodo.23029439](https://doi.org/10.5281/zenodo.23029439) |
| Operating systems | Linux. Developed on RHEL 9.4. POSIX only, not tested on macOS or Windows |
| Programming language | Python 3.9 |
| Other requirements | PyTorch 1.11.0 with CUDA 11.3, PyTorch Geometric 2.0.4, torch-scatter 2.0.9, RDKit 2024.03.6, NumPy 1.23.1, Open Babel 3.1.1, and one NVIDIA GPU for training and sampling. Evaluation additionally needs AutoDock Vina 1.2.2, Meeko 0.6.1, SMINA, GNINA 1.1, PLIP 3.0.0, PoseCheck 1.3.1 with ProLIF 2.2.0, and PoseBusters 0.6.5, across three more environments |
| License | MIT, except `gated_energy_diffusion/utils/reconstruct.py` which is GPL-2.0. See NOTICE.md |
| Restrictions for non-academic use | none |

## Install

```bash
conda env create -f envs/environment-train.yml
conda activate ged-train
pip install -e .
```

The three evaluation environments are separate and must stay that way. PoseBusters and
PoseCheck pull in two different RDKits, and both contribute numbers to the tables. See
`envs/README.md`.

## Reproduce the tables

```bash
jupyter notebook reproduction.ipynb
```

Section 1 prints the tables from the summary CSVs in `results/`. It needs pandas and takes
seconds. Section 2 recomputes the chemistry, the Vina statistics and the diversity from the
generated molecules, which needs the data archive:

```bash
python tools/download_archive.py --doi 10.5281/zenodo.23029439
```

## Train

```bash
python bin/train.py --config configs/train.yml --tag myrun
python bin/train.py --config configs/train.yml --tag myrun --resume logs_diffusion/train_myrun
```

The released checkpoint is iteration 1,195,000 at batch size 2 with 2 accumulation steps,
which is on the order of a week on one 24 GB card. `--train_report_iter` and `--save_freq`
control how often it logs and checkpoints.

Training needs the processed LMDB `data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb`,
a sparse file that reports 4.07 GB and holds about 1.45 GB of data. It carries the Vina
affinity labels the objective reads, and the split indexes its byte-sorted key order, so use
that file rather than rebuilding one. `data/README.md` has the detail.

The interaction-energy weight ramps as `w_m = 0.05 * min(m / 200000, 1)` over optimizer
iterations. The energy is evaluated at the predicted clean coordinates against protein
coordinates perturbed by N(0, 0.1^2), so the minimised quantity is the pair potential
convolved with a 0.1 A Gaussian.

The chemistry gates are mandatory. If the per-atom XS flags are absent the run fails with an
assertion instead of falling back to an element-level approximation, which would change
per-atom gradient directions for 41% of atoms.

## Sample

```bash
bash bin/sample_local.sh --ckpt archive/chiediff_weights_iter1195000.pt --pockets 0-4
```

It writes to `results/generated` by default, which is what `bin/measure.sh --base` expects.
About an hour per pocket at 100 molecules on a 24 GB card, sequential and resumable. Only
`head1_only` guidance is implemented. The two channels are asymmetric: positions use
grad log A-hat and atom types use grad A-hat, matching the sampling algorithm in the paper.

## Measure

```bash
bash bin/measure.sh --pt archive/chiediff_generated_molecules.pt --tag chiediff --pockets 0-4
bash bin/measure.sh --base results/samples --tag myrun --pockets 0-4
```

Runs docking, interaction profiling and the physical checks sequentially on one machine, and
skips any step whose output already exists. `--pockets` defaults to 0-4 because the full
benchmark is 100 pockets through five instruments, which is days on one workstation.

## Notes on running this

**The sampler does not reproduce individual molecules.** The affinity position gradient is a
second backward pass over a graph the first pass retained, and re-traversing it accumulates in
a different floating-point order each time, about 2.4e-07. A thousand stochastic steps at
guidance scale 25 turn that into a different molecule, on CPU at one thread as well as on a
GPU. The archived molecule set is authoritative for molecular identity, and a fresh run
reproduces the statistics rather than the molecules. The atom-count distribution is exactly
reproducible, being drawn before any gradient.

**`OMP_NUM_THREADS=1` is required for bitwise reproducibility** of anything else. Two
identical sampling runs differ by 10 to 19 atom types at default GPU threading and by 6 to 7
at eight threads.

**The released 100 molecules per pocket came from three chunked invocations** of 24, 2 and 76
through `--start_index`, at seed 42. A single 100-molecule run at the same seed gives a
different molecule set.

**Twelve receptors encode their chain copies only in the segID column**, which Biopython and
therefore PoseCheck ignores, leaving a partial receptor. Those pockets must be measured
against the chain-remapped copies. `scripts/make_chainfix_receptors.py` regenerates them and
the archive ships them.

**ProLIF is pinned to 2.2.0.** Version 2.1.0 changed the hydrophobic SMARTS, and the same
crystal ligands give 2.04 hydrophobic contacts per molecule under the older definition against
1.44 under the newer one.

**Cohorts.** 9,999 molecules sampled, 9,857 reconstructed and exported to SDF, 8,340
docking-successful, 8,069 in the strict pose cohort. Each table states its own n.

## Built on

The model package derives from [TargetDiff](https://github.com/guanjq/targetdiff) and
[KGDiff](https://github.com/CMACH508/KGDiff), both MIT, and the coordinate-to-molecule
reconstruction comes from [liGAN](https://github.com/mattragoza/liGAN), which is GPL-2.0.
`NOTICE.md` has the per-file provenance.

## Licensing

MIT, with one GPL-2.0 file that the evaluation path depends on. Per-file provenance and the
upstream projects are in `NOTICE.md`.

## Citation

`CITATION.cff`.

## Authors

Hyoungjoon Park, Seungyeon Choi, Hwanhee Kim, Seungyong Lee, Yoonju Kim, Taeil Noh,
Minjae Lee, and Sanghyun Park (corresponding: sanghyun@yonsei.ac.kr).

Department of Computer Science, Yonsei University, 50 Yonsei-ro, Seodaemun-gu,
Seoul 03722, Republic of Korea.
