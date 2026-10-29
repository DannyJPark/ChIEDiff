# `results/` — the aggregate numbers behind the paper's tables

These are the summary files that `reproduction.ipynb` reads to print the tables. They are
aggregates, not raw data: about 230 KB of per-model rows, plus a 3.5 MB per-molecule file for
the specificity analysis, distilled from roughly 150 GB of sampled molecules, docked poses and
per-pocket measurement CSVs across nine models and five instruments.

```bash
jupyter notebook reproduction.ipynb     # reads this directory; no GPU, no downloads
```

## Integrity

`MANIFEST.sha256` covers the 22 data files here, in `sha256sum` format:

```bash
cd results && sha256sum -c MANIFEST.sha256
```

## What each file holds

| File | Holds |
|---|---|
| `comparison/f1_sbdd/main_table/main_table.csv` | Vina, Vinardo and GNINA score and dock per model, the score-dock gap, diversity |
| `comparison/f2_pose_fidelity/redocking_consistency/redocking_consistency.csv` | the score-to-dock gap and redocking RMSD per engine, one run per engine block |
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
| `diagnostics/ablation_nci/nci_summary_abl.csv` | the PLIP and ProLIF ablation arms |
| `diagnostics/ablation_nci_v2/posecheck_ablation_all.csv` | the ProLIF and PoseCheck ablation arms |
| `pose_fidelity/heavy_clash/table_inputs.csv` | PoseCheck clash counts, heavy-atom-only and hydrogen-inclusive, per table and cohort |
| `diagnostics/ablation_nci_v2/dock_success.csv` | the docking yield of each ablation arm |
| `diagnostics/ablation_nci_v2/plip_intersection.csv` | PLIP counts re-pooled on the molecules both instruments measured |
| `diagnostics/type_cf/pocket{0..4}.csv` | the atom-type substitution diagnostic |

## What is not here

| Not here | Why | Where instead |
|---|---|---|
| the generated molecules (`*.pt`) | ours is large, and the baselines' molecules were released by their projects or generated here with their code | ours in the data archive; each baseline's project is in `NOTICE.md` |
| the per-molecule measurement CSVs | intermediate; the files above are their summary | the data archive, or regenerate with `bin/measure.sh` |
| the PLIP JSONL | intermediate; the files above are their summary | regenerate with `bin/measure.sh` |
| the training LMDB | derived from CrossDocked2020, with the affinity labels the training used | the data archive. See `data/README.md` |

## Which column is which

`heavy_clash_avg` in `pose_fidelity/heavy_clash/table_inputs.csv` counts heavy-atom pairs and
is the reported PoseCheck clash column. `clash_h_incl` in the same file, and `clash_avg` in
`pose_stability_strict.csv`, count hydrogens on both sides.

`nci_summary_abl.csv` pools over every PLIP-measured molecule. `plip_intersection.csv` pools
over the molecules both PLIP and ProLIF measured, and is the one the tables use.

## Cohorts

Each file aggregates one cohort:

```
9,999 sampled -> 9,857 reconstructed and exported to SDF -> 8,340 docking-successful
                 -> 8,338 with complete scoring -> 8,069 in the strict pose cohort
```

Each table states its own n, and `reproduction.ipynb` prints it alongside each number.
