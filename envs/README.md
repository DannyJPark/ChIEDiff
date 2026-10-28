# Environments

Five, and they cannot be merged. The measurement instruments disagree about RDKit and
NumPy by two major versions, and the reported numbers depend on which version measured
them.

Every version below was confirmed by **importing the module in that environment**, not by
reading `conda list`. That distinction matters here: several of these environments carry
duplicate `.dist-info` metadata from installs that were later overwritten, so `conda list`
reports a version that is not the one on disk. For `meekovina`, `conda list` claims RDKit
2025.3.3 and pandas 1.4.1 while the interpreter actually imports 2024.03.6 and 2.1.4.
If you need to re-verify, import the module rather than trusting the metadata.

## The matrix

| File | Env name | Python | RDKit | NumPy | pandas | Key packages |
|---|---|---|---|---|---|---|
| `environment-train.yml` | `ged-train` | 3.9.23 | 2024.03.6 | 1.23.1 | 2.1.4 | torch 1.11.0+cu113, PyG 2.0.4, torch-scatter 2.0.9, scipy 1.7.3, PLIP 3.0.0, Open Babel 3.1.1, meeko 0.1.dev3, vina 1.2.2 |
| `environment-vina.yml` | `ged-vina` | 3.9.23 | 2024.03.6 | 1.23.1 | 2.1.4 | vina 1.2.2, **meeko 0.6.1**, scipy 1.13.1 |
| `environment-posecheck.yml` | `ged-posecheck` | 3.10.20 | **2026.03.4** | 2.2.6 | 2.3.3 | posecheck 1.3.1, **prolif 2.2.0**, MDAnalysis 2.9.0, biotite 1.2.0, hydride 1.2.3 |
| `environment-posebusters.yml` | `ged-posebusters` | 3.10.20 | **2026.03.3** | 2.2.6 | 2.3.3 | posebusters 0.6.5 |

Note that PoseBusters (2026.03.3) and PoseCheck (2026.03.4) are **two different RDKits**,
and both contribute cells to the reported tables, which is why the manuscript's instrument
table lists them as two rows.

## Which script runs where

| Work | Environment |
|---|---|
| training, sampling, `reproduction.ipynb` | `ged-train` |
| `group_samples.py`, `eval_export_sdf.py` | `ged-train` |
| PLIP interaction profiling | `ged-train` (PLIP 3.0.0 lives here) |
| `dock_engines.py`, which drives SMINA, Vinardo and GNINA | `ged-train` (all three are external binaries) |
| `pose_rmsd.py`, `self_redock.py`, `strain_energy.py`, `chem_props.py` | `ged-train` |
| `vina_meeko_dock.py` | `ged-vina` |
| `eval_posecheck.py`, `eval_prolif_counts.py` | `ged-posecheck` |
| `eval_posebusters.py`, `eval_pb_distance.py` | `ged-posebusters` |

## The rules that matter

**1. Call `ged-posecheck` by absolute interpreter path, never `conda activate`.**

```bash
"$CONDA_BASE/envs/ged-posecheck/bin/python" scripts/eval_posecheck.py ...
```

Activating it from a `ged-train` shell risks its RDKit 2026.03.4 shadowing the training
environment's 2024.03.6. The PLIP-side numbers were measured with 2024.03.6, and a leak
changes them.

**2. ProLIF is pinned to 2.2.0, twice.**

`conda-meta/pinned` holds `prolif ==2.2.0`, and `etc/conda/activate.d/pin_prolif.sh` exports
`PIP_CONSTRAINT` so that a `pip install` cannot pull it forward either. ProLIF is a pip
package, so the conda pin alone is not enough.

ProLIF 2.1.0 changed the hydrophobic SMARTS definition, excluding aromatic carbon bonded
to N/O/F. On the same crystal ligands, with everything else held fixed, that gives **2.04
hydrophobic contacts per molecule under the pre-2.1 definition and 1.44 under 2.1+**.
Upgrading ProLIF therefore invalidates every ProLIF column in every table. If it must ever
be upgraded, re-measure every model's ProLIF arm and regenerate the whole table, because
partially re-measuring mixes two definitions inside one table.

For the same reason, the ProLIF counts here must not be cross-cited against papers that
used a different ProLIF version. Use the PLIP arm for cross-paper comparison.

**3. meeko is 0.1.dev3 in `ged-train` and 0.6.1 in `ged-vina`.**

`ged-train`'s 0.1.dev3 must not be upgraded, because the docking path depends on that pin.
The RMSD work needs 0.6.1's atom-index round-trip, which is why it lives in a separate
environment rather than being upgraded in place. Compare poses with RDKit `CalcRMS`, not by
atom index.

**4. A batch script activates conda itself and never runs `conda init`.**

Where conda auto-initialisation is disabled in the shell profile, a fresh non-interactive
shell has no `conda` on `PATH`. Every launcher therefore does:

```bash
source "${CONDA_BASE:-$HOME/anaconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-ged-train}"
```

Do not substitute `conda init`. It rewrites the shell profile and can re-enable that
auto-initialisation.

## Creating them

```bash
conda env create -f envs/environment-train.yml
conda activate ged-train
pip install -e .          # installs the gated_energy_diffusion package itself
```

`torch`, `torch-geometric` and `torch-scatter` are pinned in the YAML against the CUDA 11.3
build line. On a different CUDA version you must substitute matching wheels. The model runs
on newer GPUs than cu113 nominally targets, an RTX 4090 working because cubins are forward
compatible within a compute-capability major, but the versions themselves should be held.

## Known version discrepancies

- An older `requirements.txt` pins `pandas==1.4.1`. The measured environment runs **2.1.4**,
  which is what the environment files here pin. The older pin is stale.
- Open Babel's conda package is 3.1.1 while the Python module reports `3.1.0`. Both refer to
  the same install, and the module-reported number is the one recorded.
- `ged-posecheck` carries a CPU-only `torch 2.13.0`, pulled in as a PoseCheck dependency. It
  is unrelated to the training torch. Note that loading a `.pt` file under torch >= 2.6
  changes the `weights_only` default, which is why the loaders in `scripts/` pass
  `weights_only=False` explicitly.
