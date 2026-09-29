# Reproducing this work

Written for **one workstation with one GPU, and no cluster**. Slurm is an option, not a
requirement: nothing here needs it, and the launchers under `slurm/` are a convenience for
people who have it.

Three tiers, and they differ by orders of magnitude in cost. Tier 1 is the one that
reproduces the published numbers.

---

## Tier 1 — reproduce every published number (~1 minute, CPU, no downloads)

```bash
git clone <this repository> && cd <repository>
make tables
```

That is the whole thing. It reads the aggregate CSVs committed under `results/` (~40 MB),
regenerates all 22 generated tables, and checks every printed cell against its source CSV —
1767 cells. No GPU, no conda environment, no archive: the generator and the renderer use only
the Python standard library.

```bash
bash bin/check.sh      # all nine gates, no regeneration
make tables-dry        # render the tables to stdout without touching the LaTeX targets
```

**If you want to confirm the published tables are what this repository produces, you are
done.** Tiers 2 and 3 exist for reusing the model, not for checking the paper.

---

## Tier 2 — re-measure from the archived molecules (hours to days)

This re-derives the aggregate CSVs from per-molecule measurements. It needs four external
tools and four conda environments.

### 2.1 Environments

```bash
conda env create -f envs/environment-train.yml       # training, sampling, aggregation, PLIP
conda env create -f envs/environment-vina.yml        # Vina API docking
conda env create -f envs/environment-posecheck.yml   # PoseCheck + ProLIF
conda env create -f envs/environment-posebusters.yml # PoseBusters
conda activate ged-train && pip install -e .
```

Read `envs/README.md` first. Two rules there are not optional: ProLIF must stay at 2.2.0 (2.1.0
changed the hydrophobic SMARTS and moves the contact counts), and the PoseCheck environment is
called by absolute interpreter path so its RDKit cannot shadow the training environment's.

External binaries, installed from their own upstreams and put on `PATH`: **SMINA** (Vinardo
scoring) and **GNINA 1.1**. Vina and PoseBusters come in through pip in the environments above.

### 2.2 The archive

```bash
python3 tools/download_archive.py --doi 10.5281/zenodo.23029439
```

Lands in `archive/`: the trained weights, the generated molecules, our measurement outputs, and
the test receptors including the chain-remapped copies. Verified against the archive's own
`SHA256SUMS`.

If you would rather not download the receptors, regenerate them from your own CrossDocked2020
copy with `python3 scripts/make_chainfix_receptors.py`. Do not skip the chain-remapping step:
twelve receptors encode their chain copies only in the segID column, which Biopython ignores,
so measuring against the originals silently uses a partial receptor for those pockets.

### 2.3 Measure, then aggregate

```bash
bash bin/measure.sh --samples archive/samples --tag mymodel --pockets 0-4
bash bin/remeasure.sh
```

`bin/measure.sh` runs the four instruments sequentially on the local machine. It defaults to
**five pockets**, deliberately: the published benchmark is nine models × 100 pockets × ~100
molecules × four instruments, which is weeks on one workstation. Five pockets confirms the
pipeline works. Pass `--pockets 0-99` for the full set, or `--slurm` on a cluster.

Every step skips outputs that already exist, so an interrupted run resumes.

`bin/remeasure.sh` aggregates the full nine-model roster and will name the models whose
measurement outputs it cannot find. Comparing your own model against the published ones needs
their measurements too; `results/README.md` explains what ships and what does not.

---

## Tier 3 — train from scratch (days on one GPU)

```bash
conda activate ged-train
python bin/train.py --config configs/train.yml --tag myrun
```

The published checkpoint is iteration 1,195,000, at batch size 2 with 2 accumulation steps.
On a single 24 GB card that is on the order of a week. The trainer checkpoints continuously and
`--resume <run dir>` picks up where it stopped, so it survives interruption.

Two things to know before reading anything into a retrained model:

- The split has an **empty validation subset**, so the trainer validates on the test pockets.
  Checkpoint selection is therefore not independent of the evaluation set. This is shipped
  as-is because changing it would make the released weights unreproducible.
- The chemistry gates are mandatory. If the per-atom XS flags are missing the run **fails with
  an assertion** rather than falling back to an element-level approximation, which would
  change per-atom gradient directions for 41% of atoms.

### Sampling from a checkpoint

```bash
bash bin/sample_local.sh --ckpt archive/<weights file> --pockets 0-4
```

Sequential, one pocket at a time, resumable. Roughly an hour per pocket at 100 molecules on a
24 GB card.

---

## What is and is not reproducible

| | Reproducible |
|---|---|
| the published tables, from the shipped CSVs | **exactly** |
| aggregate statistics from a fresh sampling run | yes, within sampling variation |
| the per-molecule atom-count distribution | **exactly** |
| **individual molecules** | **no — see below** |
| training loss and gradients, given a fixed batch | exactly, on CPU |

The sampler does not reproduce individual molecules, and this is a property of the method as
published rather than of this release. The affinity position gradient is a second backward pass
over a graph the first retained; re-traversing it accumulates in a different float order each
time, about 2.4e-07, and 1000 stochastic steps at a guidance scale of 25 grow that into a
different molecule. Two runs of the original implementation differ by 9.18 Å and 49 atom types
— more than the original differs from this release.

So the archived molecule set is authoritative for identity, and a fresh run reproduces the
reported statistics rather than the molecules. `docs/known_limitations.md` has the measurements.

---

## If something fails

| Symptom | Cause |
|---|---|
| `ModuleNotFoundError: gated_energy_diffusion` | run from the repository root, or `pip install -e .` |
| `AssertionError: ... requires per-atom XS flags` | the training transform list is missing `FeaturizeVinaAtomTypes` — intended behaviour, not a bug |
| `ValueError: use_dual_head_sam_pl=True ...` | a config from the development tree; use `configs/train.yml` |
| hydrophobic contact counts disagree with the paper | ProLIF is not 2.2.0 |
| a pocket's interaction counts are zero with no error | receptors are not the chain-remapped copies |
| PoseBusters takes 15+ minutes on one pocket | a pathological molecule in the energy check; the worker times out per molecule |
| `sbatch: command not found` | you passed `--slurm` on a machine without it; drop the flag |
