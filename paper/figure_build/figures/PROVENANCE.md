# Figure provenance

Sixteen figures. Fifteen have a generator in this repository; one does not, and this file
exists so that exception is stated rather than discovered.

Rendered output is not committed — `make tables` and the figure scripts produce it. The one
committed image is the exception below.

| Figure | File | Generator | Needs |
|---|---|---|---|
| problem definition | `problem_definition_v2.pdf` | `problem_definition/compose_problem_figure_v2.py` | PyMOL + archived poses |
| **method overview** | `main_figure_fixed2.png` | **none — authored by hand** | — |
| redocking consistency (main) | `consistency_combined.pdf` | `consistency_combined/combine.py`, from two panels | PyMuPDF |
| ↳ panel (a) | `panels/fig2a_rsg_ecdf_vinardo.pdf` | `scripts/build_rsg_ecdf.py` | `results/` |
| ↳ panel (b) | `panels/fig2c_pose_energy_map_gnina.pdf` | `scripts/build_pose_energy_map.py` | `results/` |
| PLIP distributions | `nci_dist_plip_baselines_boxen.pdf` | `make_nci_distributions.py` | `results/` |
| ProLIF distributions | `nci_dist_prolif_baselines_boxen.pdf` | `make_nci_distributions.py` | `results/` |
| PLIP ablation | `nci_dist_plip_ablation_boxen.pdf` | `make_nci_distributions.py` | `results/` |
| ProLIF ablation | `nci_dist_prolif_ablation_boxen.pdf` | `make_nci_distributions.py` | `results/` |
| molecular properties | `property_distributions.pdf` | `scripts/build_property_tables.py` (`make_figure`) | `results/` |
| ring sizes | `ring_distributions.pdf` | `make_revision_figures.py` | `results/` |
| case study | `case_study.pdf` | `case_study/compose_case_study.py` | PyMOL + archived poses |
| energy ablation | `energy_ablation.pdf` | `make_revision_figures.py` | `paper/data/ablation_plip.csv` |
| paired marginals | `redocking_consistency_paired_full_v2.pdf` | `make_round2_consistency.py` | raw measurement tree |
| joint density | `redocking_joint_all_v2.pdf` | `make_round2_consistency.py` | raw measurement tree |
| all-model consistency | `redocking_consistency_all.pdf` | `make_revision_figures.py` | `results/` |
| token counterfactual | `type_counterfactual.pdf` | `make_revision_figures.py` | `results/diagnostics/type_cf/` |
| gate contrasts | `gate_contrasts.pdf` | `make_revision_figures.py` | `results/diagnostics/gate_within/` |

## The one figure with no generator

`main_figure_fixed2.png` is the method-overview schematic. It was authored by hand in
presentation software, so it cannot be regenerated from code. The PNG is committed and its
sha256 is recorded in `MANIFEST.sha256`, which is what guarantees the image here is the one
in the published PDF. Attempting to half-reconstruct a schematic in a plotting library would
be worse than an honestly archived asset.

## `panels/`

Two vector panels that `combine.py` places into the main consistency figure with
`show_pdf_page`, rather than replotting them together — that keeps each panel's own
typography intact at the journal's 372 pt text width.

These two panels used to sit in `unused_figures/` (now `extra/`) alongside genuinely unused
variants, and both generators carried an inline comment saying the output was "not in the
manuscript". That comment was false: the panels are half of a main-text figure. They now have
their own directory, the comments are corrected, and `bin/check.sh` fails on any file under
`figures/` that is neither referenced by a document nor declared here as a panel input — so a
figure cannot quietly become load-bearing while still labelled unused.

## `extra/`

Figure variants the manuscript does not name — violin versions of the distribution panels,
faceted versions of the energy map, alternative scoring arms. The generators write them here
deliberately so that `figures/` itself holds only what a document references. This is the
directory that was previously called `unused_figures/`; the name was accurate for these files
and misleading for the two panels, which is why the panels moved out rather than the
directory staying put.
