# `results/` — the aggregate numbers behind the paper's tables

These are the summary files that `reproduction.ipynb` reads to print the tables. They are
aggregates, not raw data: about 150 KB of per-model rows distilled from roughly 150 GB of
sampled molecules, docked poses and per-pocket measurement CSVs across nine models and five
instruments.

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
| `comparison/f1_sbdd/main_table/main_table.csv` | Vina score, minimize and dock per model, the score-dock gap, diversity |
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
| the generated molecules (`*.pt`) | ours is large, and six of the nine baselines are other papers' released generations | ours in the data archive; the baselines from their own publications |
| the per-molecule measurement CSVs and PLIP JSONL | intermediate; the files above are their summary | the data archive, or regenerate with `bin/measure.sh` |
| the per-molecule distribution caches | only the figures read them | the data archive |
| the training LMDB | derived from CrossDocked2020, with the affinity labels the training used | the data archive. See `data/README.md` |

## Two columns that are easy to confuse

**The PoseCheck clash count has two definitions.** `heavy_clash_avg` counts heavy-atom pairs
only and is the reported column. `clash_h_incl` counts hydrogens on both sides and is five to
twelve times larger. Both are in `pose_fidelity/heavy_clash/table_inputs.csv`, and
`pose_stability_strict.csv`'s `clash_avg` is the hydrogen-inclusive one, so the two files do
not agree by construction.

**The PLIP ablation counts come in two cohorts.** `nci_summary_abl.csv` pools over every
PLIP-measured molecule. `plip_intersection.csv` re-pools over the molecules both PLIP and
ProLIF measured, which is the comparison the paper makes. The two disagree, and neither is
wrong on its own terms.

## A caution about denominators

Each file aggregates one cohort, and the cohorts are not interchangeable:

```
9,999 sampled -> 9,857 reconstructed and exported to SDF -> 8,340 docking-successful
                 -> 8,338 with complete scoring -> 8,069 in the strict pose cohort
```

The attrition is not independent of the model: fragmented molecules fail docking, and the
models that fragment most lose most.
Each instrument also reports its own n, which can be smaller than the chain step it sits on,
because an instrument drops a molecule it cannot read. PoseBusters, for instance, reports
9,782 exported and 8,271 docking-successful for this work against the 9,857 and 8,340 above.
Read the n printed with a table rather than assuming the chain. A number taken from one of these files without its
cohort is not comparable with a number from another. Every table in the paper states which
cohort it aggregates, and `reproduction.ipynb` prints the count alongside each number.
