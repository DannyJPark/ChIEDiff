# Reproducing this work

One workstation with one GPU.

## Tables, from the shipped summaries

```bash
jupyter notebook reproduction.ipynb
```

Minutes, CPU, no downloads. Section 2 additionally needs the archive.

## Re-measuring from the molecules

```bash
conda env create -f envs/environment-vina.yml
conda env create -f envs/environment-posecheck.yml
conda run -n ged-posecheck pip install --no-deps posecheck==1.3.1
conda env create -f envs/environment-posebusters.yml
python tools/download_archive.py --doi 10.5281/zenodo.23029438
bash bin/measure.sh --pt archive/chiediff_generated_molecules.pt --tag chiediff --pockets 0-4
```

SMINA (the 2019-10-15 static build, <https://sourceforge.net/projects/smina/files/>) and GNINA 1.1
(<https://github.com/gnina/gnina/releases/tag/v1.1>) are single binaries. Put them on `PATH` or set
`$SMINA` and `$GNINA`.
Vinardo is SMINA with another scoring flag. GNINA is pinned at 1.1 because 1.3 and above need
a newer glibc than many cluster images carry.

| Step | Script | Output |
|---|---|---|
| per-pocket SDFs | `eval_export_sdf.py` | `eval_out/<tag>/sdf/` |
| Vina score, minimize, dock | `vina_meeko_dock.py` | `eval_out/<tag>/vina_meeko/` |
| SMINA, Vinardo, GNINA | `dock_engines.py` | `eval_out/<tag>/{smina,smina_vinardo,gnina}/` |
| pose RMSD | `pose_rmsd.py` | `eval_out/<tag>/pose_rmsd.csv` |
| PoseCheck, ProLIF | `eval_posecheck.py`, `eval_prolif_counts.py` | `eval_out/<tag>/{posecheck,prolif_counts}/` |
| PoseBusters | `eval_posebusters.py`, `eval_pb_distance.py` | `eval_out/<tag>/{posebusters,pb_distance}/` |
| PLIP | `plip_interactions.py` | `eval_plip/<tag>/` |
| strain, properties | `strain_energy.py`, `chem_props.py` | `eval_out/<tag>/` |

`--pockets` defaults to 0-4. The full benchmark is 100 pockets through five instruments, days
on one workstation. On a re-run, the SMINA, Vinardo, GNINA and PoseBusters steps skip a pocket
whose output exists, and the other steps run again.

## Training

```bash
conda activate ged-train
python bin/train.py --config configs/train.yml --tag myrun --save_freq 100
```

Needs `data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb` from the archive. It
carries the Vina affinity labels the objective reads, and the split indexes its byte-sorted
key order, so use that file rather than rebuilding one.

`--save_freq` defaults to the config's `val_freq` (1000), so a run shorter than that leaves
only the iteration-0 checkpoint without it.
The split's validation subset is empty and the trainer validates on the test subset; leave it
as it is, since the released weights were trained that way. The chemistry gates are
mandatory: without the per-atom XS flags the run fails with an assertion.

### Sampling

```bash
bash bin/sample_local.sh --ckpt archive/chiediff_weights_iter1195000.pt --pockets 0-4
python scripts/group_samples.py --base results/generated --out results/myrun.pt
```

## What reproduces

| | |
|---|---|
| the published tables, from the summaries or the molecules | exactly |
| aggregate statistics from a fresh sampling run | within sampling variation |
| the per-molecule atom-count distribution | exactly |
| training loss and gradients on a fixed batch | exactly, CPU at one thread |
| individual molecules from a fresh sampling run | no |

The affinity position gradient is a second backward pass over a graph the first retained, and
re-traversing it accumulates in a different floating-point order each time, about 2.4e-07.
A thousand stochastic steps at guidance scale 25 grow that into a different molecule.

## If something fails

| Symptom | Cause |
|---|---|
| `ModuleNotFoundError: gated_energy_diffusion` | run from the repository root, or `pip install --no-deps -e .` |
| `AssertionError: ... requires per-atom XS flags` | the transform list is missing `FeaturizeVinaAtomTypes` |
| `ValueError: use_dual_head_sam_pl=True` | a configuration this release does not implement |
| `receptor not found` | run from the repository root; `manifest.csv` paths are relative to it |
| `receptors not found: data/test_set_chainfix` | run the downloader, or `scripts/make_chainfix_receptors.py` |
| hydrophobic counts disagree | ProLIF is not 2.2.0 |
| a pocket's interaction counts are zero | receptors are not the chain-remapped copies |
| `SKIPPED smina: not on PATH` | install it, or set `$SMINA` |
| PoseBusters runs very long on one pocket | a pathological molecule; the worker times out per molecule |
