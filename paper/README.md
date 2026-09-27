# `paper/` — the table harness

**`main.tex` and `supplementary.tex` here are not the manuscript.** The manuscript source is
not shipped before acceptance. These two files exist so the table generator has somewhere to
inject into: each carries only the `BEGIN/END AUTO` markers for the tables that
`float_registry.json` places in it.

```bash
make tables        # inject, then check every cell against its source CSV
make tables-dry    # render to stdout, write nothing
```

On acceptance the two harness files are replaced by the real manuscript, which carries the
same markers in the same places. Nothing about the generator, the registry or the gates
changes.

## Why the directory is still called `paper/`

`tables/build_revision_tables.py` resolves `ROOT = Path(__file__).resolve().parents[2]` and
then hardcodes `paper/data`, `paper/float_registry.json` and `paper/main.tex` as string
constants, and shells out to `scripts/csv_to_latex.py` with `cwd=ROOT`. The figure scripts
independently do `sys.path.insert(0, PAPER / "tables")`. Renaming the directory means editing
those constants across several files, where some paths are pinned by sha256 elsewhere — and a
broken hash-pinned path fails in a way that looks like a data error.

## Layout

| Path | What |
|---|---|
| `main.tex`, `supplementary.tex` | harness targets, 5 and 17 table markers |
| `float_registry.json` | which document each float belongs in; the single source of that truth |
| `tables/build_revision_tables.py` | the generator: derives 12 CSVs, emits 22 tables |
| `data/` | derived CSVs (regenerated, gitignored) plus the tracked inputs |
| `figure_build/` | the 15 scripted figure generators |
| `figure_build/figures/` | `main_figure_fixed2.png`, the one figure with no generator |

## `data/`: what is tracked and what is not

Tracked, because they are inputs:

- `score_distributions.csv.gz`, `round2_vinardo_pairs.csv.gz` — per-molecule caches
- `round2_cohort_audit.csv` — regenerating this one needs the raw measurement tree, so it is
  a Tier-2 artifact, unlike the twelve that rebuild from `results/` at Tier 1
- `figure_manifest.json`, `nci_distribution_manifest.json`, `round2_figure_manifest.json`,
  `score_provenance.json`

Not tracked, because `make tables` regenerates them from `results/` every time:
`coverage.csv`, `main_quality.csv`, `main_chemistry.csv`, `main_ablation.csv`,
`pose_table.csv`, `properties.csv`, `distributions.csv`, `posebusters.csv`,
`ablation_ON.csv`, `ablation_OFF.csv`, `ablation_plip.csv`, `specificity_audit.csv`,
`manifest.json`, `table_specs.json`.

## The gates

`scripts/check_paper_tables.py` compares every printed cell against its source CSV —
1767 cells across the 22 tables. `scripts/check_rosters.py --shipped` pins each table's
ordered row list against a recorded fixture and cross-checks it against the registry.
`bin/make_tables.sh` runs both, so a table cannot be built from altered inputs or drift out
of agreement with the registry silently.

## Figures

Fifteen of the sixteen manuscript figures have a generator under `figure_build/`;
`figures/PROVENANCE.md` records which script makes which file. The exception is the
method-overview schematic, authored by hand in presentation software. Two further figures
need PyMOL and the archived poses to regenerate.
