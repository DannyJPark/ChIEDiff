# Attribution and third-party licences

This repository is MIT-licensed. `LICENSE` holds the plain MIT text and nothing else, so
that automated licence detection recognises it; the exceptions live here instead. The one
that matters: `gated_energy_diffusion/utils/reconstruct.py` is GPL-2.0 and keeps its own
header. Read this file before redistributing.

Every licence below was verified against a primary source, not against a README badge — two
of them are easy to get wrong, and both traps are recorded so the next person does not repeat
the check.

## Per-file provenance

| File in this repository | Upstream | Licence | Verified from |
|---|---|---|---|
| `gated_energy_diffusion/utils/reconstruct.py` | [liGAN](https://github.com/mattragoza/liGAN) `fitting.py` | **GPL-2.0** | the file's own header + GitHub API |
| `gated_energy_diffusion/utils/vina_rules.py` | [KGDiff](https://github.com/CMACH508/KGDiff) | MIT, Copyright (c) 2023 CMACH508 | upstream `LICENSE` |
| `gated_energy_diffusion/models/uni_transformer.py` | [TargetDiff](https://github.com/guanjq/targetdiff) | MIT, Copyright (c) 2023 Jiaqi Guan | upstream `LICIENCE` |
| `gated_energy_diffusion/models/common.py` | [TargetDiff](https://github.com/guanjq/targetdiff) | MIT, Copyright (c) 2023 Jiaqi Guan | upstream `LICIENCE` |
| `gated_energy_diffusion/models/score_model.py` | derived from TargetDiff / KGDiff, extended here | MIT | — |
| `gated_energy_diffusion/datasets/`, `gated_energy_diffusion/utils/{data,misc,train,transforms}.py` | TargetDiff lineage | MIT | — |
| `gated_energy_diffusion/utils/warmup.py` | [pytorch-gradual-warmup-lr](https://github.com/ildoonet/pytorch-gradual-warmup-lr) via TargetDiff | MIT, Copyright (c) 2019 **Ildoo Kim** | the file's own header |
| `gated_energy_diffusion/utils/vina_types.py` | atom typing ported from AutoDock Vina `src/lib/atom_constants.h` | Apache-2.0 (Vina 1.2); GPL-2.0 in Vina 1.1.2 and earlier | upstream source tree |
| `gated_energy_diffusion/utils/evaluation/{analyze,atom_num,atom_num_config,eval_atom_type,eval_bond_length,eval_bond_length_config,scoring_func}.py` | [TargetDiff](https://github.com/guanjq/targetdiff) | MIT, Copyright (c) 2023 Jiaqi Guan | upstream `LICIENCE` |
| `gated_energy_diffusion/utils/evaluation/sascorer.py`, `fpscores.pkl.gz` | RDKit Contrib `SA_Score` | **BSD-3-Clause**, Copyright (c) 2013 Novartis | header at `sascorer.py` end; `fpscores.pkl.gz` is byte-identical to RDKit's |
| `scripts/data_preparation/clean_crossdocked.py` | [TargetDiff](https://github.com/guanjq/targetdiff) | MIT, Copyright (c) 2023 Jiaqi Guan | upstream `LICIENCE` |
| `configs/posebusters_dock_1thread.yml` | [PoseBusters](https://github.com/maabuu/posebusters) 0.6.5 `config/dock.yml`, with `num_threads: 1` | **BSD-3-Clause** | upstream `LICENSE` |
| `bin/train.py`, `bin/sample.py` | entry points, partly derived from TargetDiff's | MIT | — |
| `data/crossdocked_pocket10_pose_split.pt` | the train/test split published with [TargetDiff](https://github.com/guanjq/targetdiff), over [CrossDocked2020](https://bits.csb.pitt.edu/files/crossdock2020/) | split MIT (TargetDiff); underlying data **CC0 1.0** | upstream `LICIENCE`; dataset `LICENSE` |
| `data/test_pocket_order.json` | the ligand filenames of that split's test subset, in split order | as the split above | derived here from the split |

## The GPL-2.0 file

`gated_energy_diffusion/utils/reconstruct.py` is GPL-2.0 and is **required** for the
evaluation pipeline: it turns sampled coordinates into molecules, so every SDF, every
measurement and therefore every table depends on it. It retains its own licence header.

Note that upstream KGDiff ships this same GPL-2.0 file inside an MIT-licensed repository.
We follow that arrangement rather than vendoring a re-implementation, because a
re-implementation would change reconstruction behaviour and therefore change the published
numbers. Readers who need a GPL-free pipeline should replace this one file and re-measure.

## Two verification traps

**TargetDiff's licence file is misspelled `LICIENCE`.** GitHub's licence detector returns
`null` for the repository and `raw.githubusercontent.com/.../LICENSE` returns 404. The
README shows an MIT badge, but a badge is not a grant. The real file is at
`https://raw.githubusercontent.com/guanjq/targetdiff/main/LICIENCE` and contains the
standard MIT text with `Copyright (c) 2023 Jiaqi Guan`.

**CrossDocked2020 is CC0, not GPL-2.0.** Web search summaries state GPL-2.0, apparently
conflating the dataset with gnina (which is Apache-2.0). The dataset's own `LICENSE` file
is the CC0 1.0 Universal Public Domain Dedication, which places no restriction on
redistribution or commercial use. This is why the archived receptor sets can be
redistributed, and why the dataset satisfies the journal's requirement that data be under
a Creative Commons licence.

## External tools (invoked, not redistributed)

These are installed by the user from their own upstreams. Versions are pinned in `envs/`
because the measurements depend on them, not for licensing reasons.

Two of them also appear in the per-file table above, because something derived from them is
redistributed here: the atom typing in `utils/vina_types.py` is ported from AutoDock Vina's
source, and `configs/posebusters_dock_1thread.yml` is PoseBusters' own `dock.yml` with one
value changed.

| Tool | Version | Licence |
|---|---|---|
| AutoDock Vina (Python API) | 1.2.2 | Apache-2.0 |
| SMINA (Vinardo scoring) | 2019-10-15 | GPL-2.0 |
| GNINA | 1.1 | Apache-2.0 |
| PLIP | 3.0.0 | GPL-2.0 |
| PoseCheck | 1.3.1 | MIT |
| ProLIF | 2.2.0 | Apache-2.0 |
| PoseBusters | 0.6.5 | BSD-3-Clause |
| RDKit | 2024.03.6 / 2026.03.3 / 2026.03.4 | BSD-3-Clause |
| Open Babel | 3.1.1 | GPL-2.0 |

Licences in this table are stated as commonly published for those projects and were not
each re-verified against a primary source; only the redistributed files above were.

## Redistributed data

| What | Where | Licence |
|---|---|---|
| the split and the test pocket order | this repository, `data/` | see the per-file table above |
| the summary result files | this repository, `results/` | CC-BY-4.0, as the data archive below |
| trained weights, generated molecules, measurement outputs, evaluation receptors | [10.5281/zenodo.23029439](https://doi.org/10.5281/zenodo.23029439) | CC-BY-4.0, receptors CC0 |

CC-BY-4.0 requires attribution when the data are reused. Cite the archive by its DOI.

## Restrictions on non-academic use

The code is MIT, with the one GPL-2.0 file noted above.

`results/` also holds rows measured from generated molecules that other projects released.
Those projects' own terms govern their released molecules, and they are not uniform. See the
per-baseline table below.
