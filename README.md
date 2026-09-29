# ChIEDiff — chemistry-gated interaction-energy diffusion for structure-based 3D molecular generation

> **The model name is provisional.** In code it exists only as `MODEL_NAME` in
> `gated_energy_diffusion/__init__.py`; the package name is function-based and will not
> change. See [Renaming the model](#11-renaming-the-model) for the complete rename surface.

## 1. What this is

A pocket-conditioned 3D diffusion model that adds a **chemistry-gated, Vina-derived
intermolecular energy** to the training objective, evaluated at the predicted clean
coordinates, and steers sampling with a learned **affinity-head gradient** (KGDiff-style,
`head1_only`, coordinate scale `s_x = 25`, atom-type scale `s_v = 100`).

Hydrophobic and donor–acceptor gates are computed from ground-truth training-complex
chemistry and held fixed, so the energy branch sends no gradient into the atom-type head.
The repository also contains the evaluation pipeline that produced every table in the
accompanying manuscript.

**Scope.** This is the single configuration the paper reports. The loss variants explored
during development (PIGNet-style, PGDiff-style, Lennard-Jones/PIDiff-style), the inactive
dual-head guidance, and the one-off diagnostic experiments are deliberately **not** here.

## 2. Availability and requirements

| Field | Value |
|---|---|
| **Project name** | ChIEDiff (provisional; package `gated_energy_diffusion`) |
| **Project home page** | https://github.com/DannyJPark/ChIEDiff |
| **Archived version** | [10.5281/zenodo.23000086](https://doi.org/10.5281/zenodo.23000086) — this release, v1.0.0. The version-independent concept DOI, which always resolves to the latest release, is [10.5281/zenodo.23000085](https://doi.org/10.5281/zenodo.23000085) |
| **Data & weights archive** | [10.5281/zenodo.23029439](https://doi.org/10.5281/zenodo.23029439) — trained weights, generated molecules, our measurement outputs, and the evaluation receptors (212 MB). Concept DOI [10.5281/zenodo.23029438](https://doi.org/10.5281/zenodo.23029438) |
| **Operating system(s)** | Linux. Developed and tested on RHEL 9.4 (kernel 5.14.0). POSIX only; not tested on macOS or Windows |
| **Programming language** | Python 3.9 · Bash · LaTeX |
| **Other requirements** | PyTorch 1.11.0 + CUDA 11.3, PyTorch Geometric 2.0.4, torch-scatter 2.0.9, RDKit 2024.03.6, NumPy 1.23.1, Open Babel 3.1.1. Evaluation additionally needs AutoDock Vina 1.2.2, SMINA (Vinardo), GNINA 1.1, PLIP 3.0.0, PoseCheck 1.3.1 + ProLIF 2.2.0, PoseBusters 0.6.5 — **five mutually incompatible conda environments**, see [`envs/README.md`](envs/README.md). One NVIDIA GPU for training and sampling. **Regenerating all 22 tables from the shipped CSVs needs only the Python 3.9 standard library and a TeX installation** — no GPU, no downloads |
| **License** | MIT, except `gated_energy_diffusion/utils/reconstruct.py` which is GPL-2.0. See [`NOTICE.md`](NOTICE.md) |
| **Restrictions for non-academic use** | None |

## 3. Quick start

Three tiers, with their honest costs.

No cluster required. Everything runs on one workstation with one GPU; Slurm is an option,
not a prerequisite.

```bash
# Tier 1 — reproduce every number in the manuscript.  ~1 min, CPU, no downloads, no conda.
make tables

# Tier 2 — re-measure from the archived molecules.  Hours to days, four conda environments.
python3 tools/download_archive.py --doi 10.5281/zenodo.23029439
bash bin/measure.sh --samples archive/samples --tag mymodel --pockets 0-4
bash bin/remeasure.sh

# Tier 3 — train, or sample from the released weights.  Days on one GPU.
python bin/train.py --config configs/train.yml --tag myrun
bash bin/sample_local.sh --ckpt archive/<weights file> --pockets 0-4
```

Tier 1 is the one a reviewer runs, and it is sufficient to check the paper: it reads the
aggregate CSVs committed under `results/` (~40 MB), regenerates all 22 generated tables, and
verifies all 1767 printed cells against their source CSVs. It needs only the Python standard
library and a TeX installation.

Tiers 2 and 3 default to **five pockets**, not 100. The full benchmark is nine models × 100
pockets × ~100 molecules across four instruments, which is weeks on one machine; five pockets
confirms the pipeline runs. `docs/reproduce.md` has the detail, including a symptom table.

## 4. Repository layout

```
gated_energy_diffusion/   the model package — the only substantially rewritten code
bin/                      entry points: train.py, sample.py, and the pipeline drivers
configs/                  train.yml (the paper configuration) and the model registry
scripts/                  measurement workers, aggregators, shared libraries, gates
tables/                   the table harness: generator + LaTeX targets + derived CSVs
figure_build/             figure generators and figure provenance
results/                  the ~39 MB of aggregate CSVs the tables are built from
data/                     the CrossDocked2020 train/test split
envs/                     five conda environment files and why they cannot be merged
slurm/                    partition-agnostic batch launchers
tools/                    archive download, provenance and placeholder/name guards
docs/                     reproduction guide, environments, provenance, limitations
```

`scripts/` and `results/` keep the path names the original pipeline used. That is
deliberate: the table generator and several gates hardcode `results/...` paths as string
constants and pin some of them by sha256, so renaming would mean editing ~30 constants
across ~15 files, where every edit is a chance to break a hash-pinned path and have the
failure look like a data error. This release gets clean by **subtraction** — roughly 250
scripts down to 60 — not by rearrangement.

## 5. Environments

Five, because the instruments disagree about RDKit and NumPy and must not be merged.
Full table and rationale in [`envs/README.md`](envs/README.md). Two rules matter:

- **PoseCheck is invoked by absolute interpreter path, never `conda activate`.** Its RDKit
  2026.03.4 must not leak into the training environment's 2024.03.6.
- **ProLIF is pinned to 2.2.0.** Version 2.1.0 changed the hydrophobic SMARTS definition;
  the same crystal ligands give 2.04 hydrophobic contacts per molecule under the
  pre-2.1 definition and 1.44 under 2.1+. Our ProLIF counts are therefore not
  comparable with values from papers that used a different ProLIF.

PoseBusters (RDKit 2026.03.3) and PoseCheck (RDKit 2026.03.4) are two different RDKits and
both contribute cells to the tables.

## 6. Reproducibility notes

These are the traps. Read them before drawing conclusions from a re-run.

**6.1 `OMP_NUM_THREADS=1` is required for bitwise reproducibility.** Two identical sampling
runs differ by 10–19 atom types at default GPU threading and by 6–7 at `OMP=8`. Only at one
thread are they bit-identical.

**6.2 The split is `train=99990`, `val=0`, `test=100`.** Because `val` is empty the trainer
uses the test pockets as its validation set, so **checkpoint selection is not independent of
the evaluation pockets.** The manuscript states this; it is repeated here because it is a
property of the shipped weights, not just of the paper.

**6.3 The published 100 molecules per pocket came from three chunked invocations**
(24 + 2 + 76) via `--start_index`, at a fixed seed of 42. A single 100-molecule run at the
same seed yields a **different molecule set**. The archived samples are authoritative for
molecule identity; a fresh run reproduces the statistics, not the identities.

**6.4 The sampler does not reproduce individual molecules, even on CPU at one thread with
identical weights, inputs and seeds.** The affinity position gradient is a second backward
pass over a graph the first retained, and re-traversing it accumulates in a different float
order each time — about 2.4e-07, which 1000 stochastic steps at a guidance scale of 25 grow
into a different molecule. Running the same code twice gives different molecules; the
original implementation's own comment says so. The archived molecule set is authoritative
for identity, and a fresh run reproduces the reported statistics, not the molecules. The
atom-count distribution *is* exactly reproducible, being drawn before any gradient.

Separately, the GPU that produced the published samples is not recorded: no Slurm log
survives and the sampler logged only its arguments. Both launchers here now log
`nvidia-smi -L`.

**6.5 Cohort chain.** 10,000 attempted → 9,857 exported to SDF → 8,340 docking-successful
(15.4% fragmented) → 8,338 with complete scoring → 8,069 in the strict pose cohort. Every
table states which of these it aggregates, and they are not interchangeable denominators.

**6.6 Canonical pocket order keys on the full ligand filename, never the pocket directory.**
Seven CrossDocked sites appear twice under one directory with different reference ligands.
A "regroup by order of appearance" bug once paired 77 pockets with the wrong receptor;
`scripts/check_pockets_equivalence.py` is the standing gate against its return.

**6.7 Twelve receptors encode their chain copies only in the segID column,** which
Biopython (and therefore PoseCheck 1.3.1) ignores. Those pockets must be measured against
the chain-remapped receptors. `scripts/make_chainfix_receptors.py` regenerates them, and
the archive ships them directly.

## 7. Reproducing the tables

`make tables` goes from the 19 shipped source CSVs to 22 generated LaTeX tables, checking
each printed cell against its CSV. Two further tables in the manuscript are hand-written
(the notation table and the instrument-version table) and are not generated.

The manuscript sources are **not** in this repository before acceptance; the tables are
injected into standalone harness documents under `tables/` instead. On acceptance the
harness targets are swapped for the manuscript files, which is a one-line change in
`tables/float_registry.json`.

## 8. Training

```bash
python bin/train.py --config configs/train.yml --tag myrun          # local GPU
sbatch -p <partition> -q <qos> slurm/train.sbatch                   # or on a cluster
```

`--resume <run dir>` picks up where a stopped run left off, so this survives interruption.

The interaction-energy weight ramps as `ω_m = 0.05 · min(m / 200000, 1)` over optimizer
iterations. That ramp lives in the training loop, not in the model. The energy is evaluated
at the predicted clean coordinates against protein coordinates perturbed by N(0, 0.1²), so
the quantity minimised is the pair potential convolved with a 0.1 Å Gaussian.

The chemistry gates are mandatory: if the per-atom XS flags are absent the run **fails with
an assertion** rather than silently falling back to an element-level approximation. That
fallback existed in development and changes per-atom gradient directions for 41% of atoms.

## 9. Sampling

```bash
bash bin/sample_local.sh --ckpt archive/<weights file> --pockets 0-4
sbatch -p <part> -q <qos> --array=0-99 slurm/sample_array.sbatch    # or on a cluster
```

Sequential and resumable locally; about an hour per pocket at 100 molecules on a 24 GB card.

Only `head1_only` guidance is implemented. The two guidance channels are deliberately
asymmetric: **positions use ∇log Â and atom types use ∇Â**, matching the manuscript's
sampling algorithm. `--seed` and `--start_index` are both exposed; see 6.3 for why the
combination matters.

## 10. Licensing and attribution

MIT, with one GPL-2.0 file that the evaluation pipeline depends on. Per-file provenance,
the upstream projects, and two licence-verification traps worth knowing about are
documented in [`NOTICE.md`](NOTICE.md).

## 11. Renaming the model

The model name appears in exactly these places:

| Site | What to change |
|---|---|
| `gated_energy_diffusion/__init__.py` | `MODEL_NAME` |
| `README.md` | title and the Project name field |
| `NOTICE.md`, `CITATION.cff` | title |
| `pyproject.toml` | the `Homepage` URL (it carries the repository name) |
| `paper/main.tex`, `paper/supplementary.tex` | `\newcommand{\method}` |

Outside the repository: the GitHub repository name and the Zenodo record title. Nothing
else references it — `tools/check_no_model_name.py` enforces that in pre-commit, so a leak
fails the commit rather than surviving to the next rename.

## 12. Known limitations

See [`docs/known_limitations.md`](docs/known_limitations.md). In short: intramolecular
strain remains high despite favourable intermolecular scores, a substantial fraction of
generated molecules are fragmented and therefore absent from the scoring cohorts (the
missingness is not independent of the quantity being compared), and the evaluation
conditions on docking eligibility throughout.

## 13. Citation

See [`CITATION.cff`](CITATION.cff).

## 14. Authors

Hyoungjoon Park, Seungyeon Choi, Hwanhee Kim, Seungyong Lee, Yoonju Kim, Taeil Noh,
Minjae Lee, and Sanghyun Park (corresponding: sanghyun@yonsei.ac.kr).

Department of Computer Science, Yonsei University, 50 Yonsei-ro, Seodaemun-gu,
Seoul 03722, Republic of Korea.
