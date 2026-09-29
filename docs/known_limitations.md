# Known limitations

Stated plainly, because several of these are easy to miss and each one bounds what a number
from this repository can support.

## Scientific limitations

### Intramolecular strain remains high

The training energy is purely **intermolecular**. It omits intramolecular and torsional
contributions and is not the complete Vina docking score. Favourable contact counts and
small redocking displacement therefore coexist with substantial internal strain: the median
PoseCheck strain for the published model is far above that of crystal ligands, and the
bond-length and bond-angle pass rates under PoseBusters are well below the best baseline's.

Larger contact counts do not establish better binding, and low redocking RMSD measures
consistency with a particular search procedure rather than correctness against an
experimental pose.

### Missingness is not independent of the quantity being compared

Every table conditions on molecules that could be docked and measured, so the failures sit
outside the denominators rather than inside them. About 15% of the published model's exported
molecules are disconnected fragments, and connectivity and docking eligibility select the
**same** set of molecules. The molecules absent from the scoring, pose and interaction
cohorts are therefore systematically the ones the energy term damaged.

This is the central caveat on every comparison in the paper. A single intersection makes
measurements comparable *within* a table; it does not recover the missing generation
denominator or remove the selection bias.

Some baseline sample files record no failed generation at all, so a connectivity rate of
100% computed from them describes what the file retains, not an unconditional success rate.
The disconnected shares are not a like-for-like ranking across all models.

### Checkpoint selection is not independent of the evaluation pockets

The split ships as `train=99990`, `val=0`, `test=100`. Because the validation subset is
empty, the trainer uses the **test** pockets for validation, and the published checkpoint
(iteration 1,195,000) was selected by validation loss computed on them.

This is shipped as-is deliberately: "fixing" the split would make the released weights
unreproducible. But it means the evaluation is not a clean held-out measurement, and no
checkpoint-selection-independent result is established by these records.

### No training-seed uncertainty

Everything is a single training run at seed 2021. Numerical rankings in the tables are
descriptive. The bootstrap intervals that do appear resample pockets, not training seeds,
and they hold the 100-ligand reference fixed, so they omit both reference-sampling and
seed uncertainty.

### The pocket-specificity analysis is exploratory

It uses a single random off-target assignment (seed 2027), asymmetric docking settings
between the on-target and off-target arms, and winsorised energies. The raw and clipped
macro-means differ materially for some models, which is itself the finding: the reported
value depends on the numerical handling. Winsorisation does not turn a failed grid
evaluation into a valid binding energy, and the magnitude-based screen shipped here is a
sensitivity analysis, not a complete failure-log audit.

## Reproducibility limitations

### The sampler is not reproducible at the level of individual molecules

This is stronger than a missing-provenance problem, and it holds on CPU, at one thread, with
identical weights, inputs and seeds. Measured directly:

| comparison | affinity position gradient |
|---|---|
| the same model instance, called twice | 2.38e-07 |
| two instances of the same class | 2.38e-07 |
| the released code against the original | 2.68e-07 |

The forward pass is bit-identical in every case. The position gradient is the second of two
backward passes over a graph the first call retained, and re-traversing that graph accumulates
in a different float order each time. Over 1000 stochastic steps with a coordinate guidance
scale of 25, that difference grows into a different molecule — the original implementation's
own source comment says exactly this.

Consequence: **running the same code twice produces different molecules.** The archived
molecule set is the authoritative record of identity; a fresh run reproduces the reported
statistics, not the molecules. This is a property of the method as published, not of this
release, and the equivalence tests in `scripts/verify_sampling_equivalence.py` are written
around it: they compare the release against the reference's own run-to-run spread rather than
demanding an identity the reference cannot deliver.

The atom-count distribution *is* exactly reproducible, because it is drawn from the
pocket-size prior before any gradient is taken.

### The GPU that produced the published samples is unrecorded

No Slurm log survives for the production sampling run, and the sampler logged only its
arguments — no node name, no device name. The launcher fans out across several GPU models,
so the published cohort may not even be a single-device product.

Consequence: **bitwise reproduction of the published molecules is not claimed.** The
equivalence tests in this repository prove that the released code matches the original code
on the same device; they cannot prove the released code reproduces the archived molecule
identities. The archived samples are authoritative for identity; a fresh run reproduces the
statistics.

Both launchers in `slurm/` now log `nvidia-smi -L`. That line is the fix for next time.

### Bitwise reproducibility needs a single thread

At default GPU threading, two identical sampling runs differ by 10–19 atom types. At
`OMP_NUM_THREADS=8`, by 6–7. Only at one thread are they bit-identical. Any claim of
sampler determinism without that setting is unfounded.

### The published molecule set cannot be reproduced by a single run

The 100 molecules per pocket came from three chunked invocations (24 + 2 + 76) via
`--start_index`, at seed 42. A single 100-molecule run at the same seed consumes the random
stream differently and yields a different molecule set.

### Two RDKits, two protonation backends

PoseBusters resolves RDKit 2026.03.3 and PoseCheck resolves 2026.03.4; both contribute cells
to the tables. Separately, protonation is **not** performed by one shared tool: PoseCheck
shells out to `hydride` (it hardcodes that command and ignores its own `reduce_path`
argument) while PLIP delegates to Open Babel. The two instruments therefore disagree on
absolute interaction counts by construction.

Within each instrument every model gets identical treatment, so model-versus-model comparison
is valid. **Absolute counts must never be compared across the two instruments**, and counts
must not be summed across them.

### ProLIF counts are version-locked

ProLIF 2.1.0 changed the hydrophobic SMARTS definition. Same crystal ligands, everything else
fixed: 2.04 hydrophobic contacts per molecule pre-2.1, 1.44 at 2.1+. Our numbers are measured
at 2.2.0 and must not be cross-cited against papers using a different ProLIF. Use the PLIP
arm for cross-paper comparison.

### Baseline generations are not redistributed

Six of the nine compared models' molecule sets are those papers' own released generations,
used unmodified. They are not in this repository or its archive; get them from their
publications. The aggregate CSVs derived from them **are** shipped, which is what lets the
tables regenerate without them.

## Repository limitations

### One figure has no generator

The method-overview schematic was authored by hand in presentation software. Its PNG ships
and is covered by the figure manifest's sha256, but it cannot be regenerated from code.
Every other figure has a generator; `figure_build/figures/PROVENANCE.md` says which.

Two further figures need PyMOL plus the archived poses to regenerate. Their generators ship;
PyMOL is an optional requirement.

### The manuscript source is not here before acceptance

Tables are injected into standalone harness documents under `tables/` rather than into the
manuscript. The cell-level gate (every printed cell equals its source CSV) still runs; what
is deferred is the check that the manuscript's own table blocks match a fresh render.

### One GPL-2.0 file inside an MIT repository

`gated_energy_diffusion/utils/reconstruct.py` is GPL-2.0 and is required for the evaluation
pipeline. It keeps its own licence header. See `NOTICE.md`; this mirrors the upstream
project's own arrangement, which is not the same as being free of tension.
