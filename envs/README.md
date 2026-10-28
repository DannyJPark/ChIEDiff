# Environments

Four, and they cannot be merged: the instruments disagree about RDKit and NumPy by two major
versions, and the reported numbers depend on which version measured them.

| File | Env | Python | RDKit | NumPy | pandas | Key packages |
|---|---|---|---|---|---|---|
| `environment-train.yml` | `ged-train` | 3.9.23 | 2024.03.6 | 1.23.1 | 2.1.4 | torch 1.11.0+cu113, PyG 2.0.4, torch-scatter 2.0.9, scipy 1.7.3, PLIP 3.0.0, Open Babel 3.1.1, meeko 0.1.dev3, vina 1.2.2 |
| `environment-vina.yml` | `ged-vina` | 3.9.23 | 2024.03.6 | 1.23.1 | 2.1.4 | vina 1.2.2, meeko 0.6.1, scipy 1.13.1 |
| `environment-posecheck.yml` | `ged-posecheck` | 3.10.20 | 2026.03.4 | 2.2.6 | 2.3.3 | posecheck 1.3.1, prolif 2.2.0, MDAnalysis 2.9.0 |
| `environment-posebusters.yml` | `ged-posebusters` | 3.10.20 | 2026.03.3 | 2.2.6 | 2.3.3 | posebusters 0.6.5 |

PoseBusters and PoseCheck carry two different RDKits and both contribute cells to the tables.

## Which script runs where

| Work | Env |
|---|---|
| training, sampling, `reproduction.ipynb` | `ged-train` |
| `group_samples.py`, `eval_export_sdf.py`, `dock_engines.py`, `pose_rmsd.py`, `strain_energy.py`, `chem_props.py`, `self_redock.py` | `ged-train` |
| PLIP (`plip_interactions.py`) | `ged-train`, where PLIP 3.0.0 lives |
| `vina_meeko_dock.py` | `ged-vina` |
| `eval_posecheck.py`, `eval_prolif_counts.py` | `ged-posecheck` |
| `eval_posebusters.py`, `eval_pb_distance.py` | `ged-posebusters` |

SMINA and GNINA are external binaries on `PATH`, or named by `$SMINA` and `$GNINA`.

## Four rules

**1. Call `ged-posecheck` by absolute interpreter path, never `conda activate`.** Its RDKit
2026.03.4 must not shadow `ged-train`'s 2024.03.6.

```bash
$CONDA_BASE/envs/ged-posecheck/bin/python scripts/eval_posecheck.py --dir eval_out/mymodel --pocket 0
```

**2. ProLIF stays at 2.2.0.** 2.1.0 changed the hydrophobic SMARTS, and the same crystal
ligands give 2.04 hydrophobic contacts per molecule under the older definition against 1.44
under the newer one. PoseCheck depends on prolif without a version bound, so the pin is
enforced in the environment file.

**3. meeko is 0.1.dev3 in `ged-train` and 0.6.1 in `ged-vina`.** The RMSD work needs 0.6.1's
atom-index round trip, which is why it lives in a separate environment.

**4. A batch script activates conda itself and never runs `conda init`.**

```bash
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate ged-train
```

## Versions

Every version above was confirmed by importing the module in that environment rather than by
reading `conda list`, which can report a version that is not the one on disk when an install
was later overwritten.
