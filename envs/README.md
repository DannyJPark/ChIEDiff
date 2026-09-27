# Environments

Five, and they cannot be merged. The measurement instruments disagree about RDKit and
NumPy by two major versions, and the published numbers depend on which version measured
them.

Every version below was confirmed by **importing the module in that environment**, not by
reading `conda list`. That distinction matters here: several of these environments carry
duplicate `.dist-info` metadata from installs that were later overwritten, so `conda list`
reports a version that is not the one on disk. For `meekovina`, `conda list` claims RDKit
2025.3.3 and pandas 1.4.1 while the interpreter actually imports 2024.03.6 and 2.1.4.
If you need to re-verify, import the module; do not trust the metadata.

## The matrix

| File | Env name | Python | RDKit | NumPy | pandas | Key packages |
|---|---|---|---|---|---|---|
| `environment-train.yml` | `ged-train` | 3.9.23 | 2024.03.6 | 1.23.1 | 2.1.4 | torch 1.11.0+cu113, PyG 2.0.4, torch-scatter 2.0.9, scipy 1.7.3, PLIP 3.0.0, Open Babel 3.1.1, meeko 0.1.dev3, vina 1.2.2 |
| `environment-vina.yml` | `ged-vina` | 3.9.23 | 2024.03.6 | 1.23.1 | 2.1.4 | vina 1.2.2, **meeko 0.6.1**, scipy 1.13.1 |
| `environment-posecheck.yml` | `ged-posecheck` | 3.10.20 | **2026.03.4** | 2.2.6 | 2.3.3 | posecheck 1.3.1, **prolif 2.2.0**, MDAnalysis 2.9.0, biotite 1.2.0, hydride 1.2.3 |
| `environment-posebusters.yml` | `ged-posebusters` | 3.10.20 | **2026.03.3** | 2.2.6 | 2.3.3 | posebusters 0.6.5 |
| `environment-figures.yml` | `ged-figures` | 3.11+ | — | — | — | matplotlib, seaborn, **pymupdf** |

Note that PoseBusters (2026.03.3) and PoseCheck (2026.03.4) are **two different RDKits**,
and both contribute cells to the published tables. The manuscript's instrument table lists
them as two rows for that reason.

## Which script runs where

| Work | Environment |
|---|---|
| training, sampling | `ged-train` |
| all `scripts/build_*.py`, `aggregate_*.py`, table generation | `ged-train` |
| PLIP interaction profiling | `ged-train` (PLIP 3.0.0 lives here) |
| SMINA / Vinardo, GNINA | `ged-train` (both are external binaries) |
| Vina Python API docking, Meeko round-trip RMSD | `ged-vina` |
| PoseCheck strain and clash, ProLIF fingerprints | `ged-posecheck` |
| PoseBusters validity checks | `ged-posebusters` |
| figure assembly (`consistency_combined/combine.py`) | `ged-figures` |

## Four rules

**1. Call `ged-posecheck` by absolute interpreter path, never `conda activate`.**

```bash
"$CONDA_BASE/envs/ged-posecheck/bin/python" scripts/eval_posecheck.py ...
```

Activating it from a `ged-train` shell risks its RDKit 2026.03.4 shadowing the training
environment's 2024.03.6. The published PLIP-side numbers were measured with 2024.03.6; a
leak silently changes them.

**2. ProLIF is pinned to 2.2.0, twice.**

`conda-meta/pinned` holds `prolif ==2.2.0`, and `etc/conda/activate.d/pin_prolif.sh` exports
`PIP_CONSTRAINT` so that a `pip install` cannot pull it forward either. ProLIF is a pip
package, so the conda pin alone is not enough.

The reason: ProLIF 2.1.0 changed the hydrophobic SMARTS definition, excluding aromatic
carbon bonded to N/O/F. On the same crystal ligands, with everything else held fixed, that
gives **2.04 hydrophobic contacts per molecule under the pre-2.1 definition and 1.44 under
2.1+**. Upgrading ProLIF invalidates every ProLIF column in every table. If it must ever be
upgraded, re-measure every model's ProLIF arm and regenerate the whole table — partially
re-measuring mixes two definitions inside one table.

The same fact means our ProLIF counts must not be cross-cited against papers that used a
different ProLIF version. Use the PLIP arm for cross-paper comparison.

**3. meeko is 0.1.dev3 in `ged-train` and 0.6.1 in `ged-vina`, on purpose.**

`ged-train`'s 0.1.dev3 must not be upgraded — the docking path depends on that pin. The
RMSD work needs 0.6.1's atom-index round-trip, which is why it lives in a separate
environment rather than being upgraded in place. Compare poses with RDKit `CalcRMS`, not by
atom index.

**4. A batch script activates conda itself and never runs `conda init`.**

Conda auto-initialisation is disabled in the shell profile on the cluster this was developed
on, so a fresh non-interactive shell has no `conda` on `PATH`. Every launcher therefore does:

```bash
source "${CONDA_BASE:-$HOME/anaconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-ged-train}"
```

Do not substitute `conda init`: it rewrites the shell profile and can re-enable the
auto-initialisation that was disabled deliberately.

## Creating them

```bash
conda env create -f envs/environment-train.yml
conda activate ged-train
pip install -e .          # installs the gated_energy_diffusion package itself
```

`torch`, `torch-geometric` and `torch-scatter` are pinned in the YAML against the CUDA 11.3
build line. On a different CUDA version you must substitute matching wheels; the model runs
on newer GPUs than cu113 nominally targets (an RTX 4090 works, because cubins are forward
compatible within a compute-capability major), but the versions themselves should be held.

## Known version discrepancies

- `requirements.txt` in the upstream working repository pins `pandas==1.4.1`. The
  environment that produced the published numbers runs **2.1.4**. The environment files here
  pin 2.1.4; the old pin was simply stale.
- Open Babel's conda package is 3.1.1 while the Python module reports `3.1.0`. Both refer to
  the same install; the recorded provenance file notes the module-reported number.
- `ged-posecheck` carries a CPU-only `torch 2.13.0`, pulled in as a PoseCheck dependency. It
  is unrelated to the training torch. Be aware that loading a `.pt` file under torch ≥ 2.6
  changes the `weights_only` default, which is why the loaders in `scripts/` try
  `weights_only=False` explicitly.

A fuller recorded snapshot, including the rationale for the two protonation backends
disagreeing, is in `analysis/nci_benchmark/env_versions.txt`.
