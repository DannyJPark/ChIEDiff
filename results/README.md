# `results/` — the aggregate data the tables are built from

**This is not the full results tree.** The working tree these files were distilled from is
about 154 GB: every sampled molecule, every docked pose, every per-pocket measurement CSV
for nine models across four instruments. What is here is the ~40 MB of *aggregates* that the
table pipeline actually reads — enough to regenerate all 22 generated tables and most of the
figures with no downloads, no GPU, and no third-party sample files.

```bash
make tables     # ~1 minute, CPU only, uses exactly what is in this directory
```

## Integrity

`MANIFEST.sha256` covers every file here in `sha256sum` format:

```bash
cd results && sha256sum -c MANIFEST.sha256
```

`bin/make_tables.sh` runs this before it generates anything, so a table can never be built
from silently altered inputs.

## What each file feeds

### The 13 sources the table generator reads by hardcoded path

| File | Feeds |
|---|---|
| `comparison/f1_sbdd/main_table/main_table.csv` | the scoring tables, the gap table, diversity |
| `comparison/f2_pose_fidelity/pose_stability/pose_stability_strict.csv` | the strict pose cohort |
| `comparison/f2_pose_fidelity/pose_stability/pose_stability_engineonly.csv` | the RMSD-complete sensitivity cohort |
| `comparison/f3_nci/nci_summary_strict.csv` | the interaction-profile tables |
| `comparison/master_table.csv` | the pocket-specificity table |
| `comparison/f1_sbdd/properties/property_overall.csv` | calculated molecular properties |
| `comparison/f1_sbdd/ring_size/ring_size_overall.csv` | ring-size distributions |
| `comparison/f1_sbdd/bond_jsd/bond_jsd_overall.csv` | bond-type Jensen–Shannon distances |
| `comparison/appendix/posebusters/pose_quality_summary_posebusters.csv` | PoseBusters pass rates and the connectivity numbers quoted in prose |
| `comparison/f2_pose_fidelity/pb_distance/pb_distance_strict.csv` | PoseBusters heavy-atom clash counts |
| `diagnostics/ablation_nci/nci_summary_abl.txt` | the PLIP ablation table |
| `diagnostics/ablation_nci_v2/posecheck_ablation_all.csv` | the ProLIF/PoseCheck ablation tables |
| `comparison/f4_delta_score/delta_score_per_mol.csv` | the specificity audit, re-aggregated with and without clipping |

### Per-molecule caches

These are needed because several tables and figures re-aggregate from molecule level rather
than reading a summary: the paired-consistency cohort audit, the specificity clipping
sensitivity, and the distribution figures.

| File | Size | Feeds |
|---|---|---|
| `pose_fidelity/master.csv` | 20 MB | one row per (model, molecule); everything pose-related joins on its `p###_m####` key |
| `comparison/f1_sbdd/properties/property_per_molecule.csv.gz` | 3.1 MB | the property distribution figure |
| `comparison/figures/rsg_ecdf_per_molecule.csv.gz` | 2.2 MB | the score-gap ECDF panel |
| `comparison/f3_nci/nci_per_molecule_strict.csv.gz` | 0.8 MB | the interaction distribution figures |
| `diagnostics/gate_within/gate_within/gate_within_per_mol.csv` | 3.2 MB | the gate-contrast diagnostic figure |
| `diagnostics/type_cf/type_cf/pocket{0..4}.csv` | 1.5 MB | the token-substitution sensitivity figure |

## What is deliberately absent

| Not here | Why | Where instead |
|---|---|---|
| the nine models' generated molecules (`*.pt`, ~620 MB) | six of them are other papers' released generations, unmodified | our own set is in the data archive; the baselines come from their own publications |
| `eval_out/` per-pocket measurement CSVs (~1.7 GB) | intermediate; the aggregates above are their summary | regenerable with `make measure` |
| `eval_plip/` PLIP JSONL (~226 MB) | same | same |
| the training LMDB (~4 GB) | derived from CrossDocked2020 | the data archive, or build it with `scripts/data_preparation/` |

## A caution about denominators

Every file here aggregates a specific cohort, and they are **not** interchangeable. The
chain is:

```
10,000 attempted → 9,857 exported to SDF → 8,340 docking-successful
                 → 8,338 with complete scoring → 8,069 in the strict pose cohort
```

`configs/models.json` records each model's counts namespaced by family *and* by `basis`
(`exported_sdf` versus `docking_successful`), and the registry gate refuses to compare
across bases. A number read out of one of these files without its cohort is not meaningful.
