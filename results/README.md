# `results/` — the aggregate numbers behind the paper's tables

These are the summary files that `reproduction.ipynb` reads to print the tables. They are
aggregates, not raw data: about 150 KB of per-model rows distilled from roughly 150 GB of
sampled molecules, docked poses and per-pocket measurement CSVs across nine models and five
instruments.

```bash
jupyter notebook reproduction.ipynb     # reads this directory; no GPU, no downloads
```

## Integrity

`MANIFEST.sha256` covers every file here in `sha256sum` format:

```bash
cd results && sha256sum -c MANIFEST.sha256
```

## What each file holds

| File | Holds |
|---|---|
| `comparison/f1_sbdd/main_table/main_table.csv` | Vina score, minimize and dock per model, the score-dock gap, diversity |
| `comparison/f2_pose_fidelity/pose_stability/pose_stability_strict.csv` | redocking RMSD on the strict pose cohort |
| `comparison/f2_pose_fidelity/pose_stability/pose_stability_engineonly.csv` | the same on the RMSD-complete cohort, as a sensitivity check |
| `comparison/f3_nci/nci_summary_strict.csv` | interaction counts per molecule, PLIP and ProLIF |
| `comparison/master_table.csv` | the pocket-specificity comparison |
| `comparison/f1_sbdd/properties/property_overall.csv` | QED, SA, LogP, TPSA, molecular weight, Lipinski |
| `comparison/f1_sbdd/ring_size/ring_size_overall.csv` | ring-size occupancy for rings of 3 to 9 atoms |
| `comparison/f1_sbdd/bond_jsd/bond_jsd_overall.csv` | bond-length and bond-type Jensen-Shannon distances |
| `comparison/appendix/posebusters/pose_quality_summary_posebusters.csv` | PoseBusters pass rates and connectivity |
| `comparison/f2_pose_fidelity/pb_distance/pb_distance_strict.csv` | PoseBusters heavy-atom clash counts |
| `comparison/f4_delta_score/delta_score_per_mol.csv.gz` | one row per molecule for the specificity analysis, which is re-aggregated with and without clipping |
| `diagnostics/ablation_nci/nci_summary_abl.txt` | the PLIP ablation arms |
| `diagnostics/ablation_nci_v2/posecheck_ablation_all.csv` | the ProLIF and PoseCheck ablation arms |
| `diagnostics/type_cf/pocket{0..4}.csv` | the atom-type substitution diagnostic |

## What is not here

| Not here | Why | Where instead |
|---|---|---|
| the generated molecules (`*.pt`) | ours is large, and six of the nine baselines are other papers' released generations | ours in the data archive; the baselines from their own publications |
| the per-molecule measurement CSVs and PLIP JSONL | intermediate; the files above are their summary | the data archive, or regenerate with `bin/measure.sh` |
| the per-molecule distribution caches | only the figures read them | the data archive |
| the training LMDB | derived from CrossDocked2020, with the affinity labels the training used | the data archive. See `data/README.md` |

## A caution about denominators

Each file aggregates one cohort, and the cohorts are not interchangeable:

```
10,000 attempted -> 9,857 exported to SDF -> 8,340 docking-successful
                 -> 8,338 with complete scoring -> 8,069 in the strict pose cohort
```

The attrition is not independent of the model: fragmented molecules fail docking, and the
models that fragment most lose most. A number taken from one of these files without its
cohort is not comparable with a number from another. Every table in the paper states which
cohort it aggregates, and `reproduction.ipynb` prints the count alongside each number.
