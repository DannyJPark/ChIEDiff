# Reproducing this work

Written for one workstation with one GPU. Nothing here needs a cluster. The launchers under
`slurm/` are a convenience for people who have one.

The three things you might want to reproduce differ by orders of magnitude in cost.

---

## The tables (minutes, CPU, no downloads)

```bash
conda env create -f envs/environment-train.yml
conda activate ged-train
pip install -e .
jupyter notebook reproduction.ipynb
```

Section 1 of the notebook prints the paper's tables from the summary CSVs under
`results/`. The figures are not regenerated here, and `docs/known_limitations.md` says which
columns of which tables need data that only the archive carries. Section 2 recomputes the chemistry, the Vina statistics, the ring-size
distribution and the diversity from the generated molecules themselves, so the summaries can
be checked against the objects they summarise. It needs the data archive:

```bash
python tools/download_archive.py --doi 10.5281/zenodo.23029439
```

If a recomputed number disagrees with its published row by more than rounding, that is a bug
worth reporting rather than reconciling by hand.

---

## Re-measuring the molecules (hours to days)

This re-derives the per-molecule measurements from the molecules. It needs three more
environments and three external binaries.

### Environments

```bash
conda env create -f envs/environment-vina.yml        # AutoDock Vina through its Python API
conda env create -f envs/environment-posecheck.yml   # PoseCheck and ProLIF
conda env create -f envs/environment-posebusters.yml # PoseBusters
```

Read `envs/README.md` first. Two rules there are not optional. ProLIF must stay at 2.2.0,
because 2.1.0 changed the hydrophobic SMARTS and moves the contact counts. And the PoseCheck
environment is called by absolute interpreter path so its RDKit cannot shadow the training
environment's.

SMINA and GNINA 1.1 are installed from their own upstreams and must be on `PATH`, or named
through `$SMINA` and `$GNINA`.

### Receptors

The archive's receptor tarball unpacks to `data/test_set` and `data/test_set_chainfix`,
which is where every measurement step looks. To regenerate the chain-remapped copies from
your own CrossDocked2020 download instead:

```bash
python scripts/make_chainfix_receptors.py --test_set data/test_set
```

Do not skip the remapping. Twelve receptors encode their chain copies only in the segID
column, which Biopython ignores, so measuring against the originals silently uses a fraction
of the receptor for those pockets and reports no error.

### Measure

```bash
bash bin/measure.sh --pt archive/chiediff_generated_molecules.pt --tag chiediff --pockets 0-4
```

Steps, in order, each skipping work whose output already exists:

| Step | Script | Output |
|---|---|---|
| per-pocket SDFs and the manifest | `eval_export_sdf.py` | `eval_out/<tag>/sdf/` |
| Vina score_only, minimize, dock | `vina_meeko_dock.py` | `eval_out/<tag>/vina_meeko/` |
| smina, Vinardo, GNINA, dock and min | `dock_engines.py` | `eval_out/<tag>/{smina,smina_vinardo,gnina}/` |
| generated-to-engine pose RMSD | `pose_rmsd.py` | `eval_out/<tag>/pose_rmsd.csv` |
| PoseCheck clashes and strain | `eval_posecheck.py` | `eval_out/<tag>/posecheck/` |
| ProLIF contact counts | `eval_prolif_counts.py` | `eval_out/<tag>/prolif_counts/` |
| PoseBusters validity | `eval_posebusters.py` | `eval_out/<tag>/posebusters/` |
| PoseBusters clash counts | `eval_pb_distance.py` | `eval_out/<tag>/pb_distance/` |
| PLIP interaction profiles | `plip_interactions.py` | `eval_plip/<tag>/` |
| MMFF94s strain, chemical properties | `strain_energy.py`, `chem_props.py` | `eval_out/<tag>/` |

`--pockets` defaults to 0-4. The full benchmark is 100 pockets of about 100 molecules through
five instruments, which is days on one workstation, so the default is there to confirm the
pipeline runs. Pass `0-99` for the whole test set.

Comparing your own model against the nine in the paper needs their measurements too. The
summaries are in `results/`, and `results/README.md` says what is there and what is not.

---

## Training from scratch (days on one GPU)

```bash
conda activate ged-train
python bin/train.py --config configs/train.yml --tag myrun
```

The released checkpoint is iteration 1,195,000 at batch size 2 with 2 accumulation steps,
roughly a week on one 24 GB card. The trainer checkpoints every `--save_freq` iterations and
defaults to the configuration's `val_freq`, so a short run needs a smaller value to write
anything. `--train_report_iter` sets how often the loss is logged, and `--resume <run dir>`
picks up where a stopped run left off.

Training needs `data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb`, the 3.8 GB
processed training set, and **that file is not in the data archive**. It cannot be rebuilt
from CrossDocked2020 with the code here, because it carries the affinity labels the objective
reads and the repository has no script that can put them back, and it cannot be substituted
either, because the split is a list of integer indices into its byte-sorted key order.

So a retrained model is not a reproduction of the released weights. `data/README.md` has the
detail, and the two tiers above do not depend on this file.

Two things to know before reading anything into a retrained model:

- The split's validation subset is empty, so the trainer validates on the test pockets.
  Checkpoint selection is therefore not independent of the evaluation set. It ships this way
  because changing it would make the released weights unreproducible.
- The chemistry gates are mandatory. If the per-atom XS flags are missing the run fails with
  an assertion rather than falling back to an element-level approximation, which would change
  per-atom gradient directions for 41% of atoms.

### Sampling from a checkpoint

```bash
bash bin/sample_local.sh --ckpt archive/chiediff_weights_iter1195000.pt --pockets 0-4
python scripts/group_samples.py --base results/generated --out results/myrun.pt
```

Sequential, one pocket at a time, resumable, about an hour per pocket at 100 molecules on a
24 GB card. `group_samples.py` turns the sampler's per-molecule files into the single grouped
file every measurement step reads, and `bin/measure.sh --base` does both in one go.

---

## What is and is not reproducible

| | |
|---|---|
| the published tables, from the shipped summaries | exactly |
| the published tables, recomputed from the molecules | exactly, to rounding |
| aggregate statistics from a fresh sampling run | yes, within sampling variation |
| the per-molecule atom-count distribution | exactly |
| training loss and gradients, given a fixed batch | exactly, on CPU at one thread |
| individual molecules from a fresh sampling run | no |

The sampler does not reproduce individual molecules, and this is a property of the method
rather than of this repository. The affinity position gradient is a second backward pass over a
graph the first pass retained, and re-traversing it accumulates in a different floating-point
order each time, about 2.4e-07. A thousand stochastic steps at guidance scale 25 grow that
into a different molecule. The archived molecule set is authoritative for molecular identity.

---

## If something fails

| Symptom | Cause |
|---|---|
| `ModuleNotFoundError: gated_energy_diffusion` | run from the repository root, or `pip install -e .` |
| `AssertionError: ... requires per-atom XS flags` | the training transform list is missing `FeaturizeVinaAtomTypes`. Intended behaviour |
| `ValueError: use_dual_head_sam_pl=True ...` | a configuration this repository does not implement. Use `configs/train.yml` |
| `no such dataset` from a launcher | the LMDB is not in `data/`. See `data/README.md` |
| `receptor not found` from a worker | `manifest.csv` holds paths relative to where the export ran. Run from the repository root |
| hydrophobic contact counts disagree with the paper | ProLIF is not 2.2.0 |
| a pocket's interaction counts are zero with no error | the receptors are not the chain-remapped copies |
| `smina not found` | install it from its upstream, or set `$SMINA` |
| PoseBusters takes very long on one pocket | a pathological molecule in the energy check. The worker times out per molecule |
