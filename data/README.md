# `data/` — what ships here, and what you have to fetch

Two small files are in the repository. The large ones are in the data archive,
[10.5281/zenodo.23029439](https://doi.org/10.5281/zenodo.23029439), because they are hundreds
of megabytes and carry their own DOI and licence.

```bash
python tools/download_archive.py --doi 10.5281/zenodo.23029439
```

That writes the archive into `data/` and `archive/` as laid out below, and checks every file
against the archive's own `SHA256SUMS`.

## In the repository

| File | What it is |
|---|---|
| `crossdocked_pocket10_pose_split.pt` | the train/validation/test split, as three lists of integer indices into the LMDB. 99,990 train, 0 validation, 100 test |
| `test_pocket_order.json` | the canonical `ligand_filename` of each of the 100 test pockets, in pocket order |

**`test_pocket_order.json` is what makes a pocket index mean something.** Pocket identity
keys on the full ligand filename and never on the pocket directory, because seven CrossDocked
sites appear twice in the test split under one directory with different reference ligands.
Grouping by directory merges those pairs and shifts every index after them, with no error and
a plausible-looking result. Every measurement step reads this file so that the SDF exports,
the docking outputs and the interaction profiles all agree on which pocket is pocket *i*.

## From the archive

| Path after download | Size | What it is |
|---|---|---|
| `data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb` | 1.4 GB | the processed training set that the released weights were trained on |
| `data/test_set/` | 82 MB | the 100 test receptors and their reference ligands |
| `data/test_set_chainfix/` | 82 MB | the same receptors with their chain copies written into the chain column |
| `archive/chiediff_weights_iter1195000.pt` | 33 MB | the released weights |
| `archive/chiediff_generated_molecules.pt` | 65 MB | the 10,000 generated molecules with their docking scores |
| `archive/measurements/` | 66 MB | the per-pocket measurement outputs behind every table |

## Why the LMDB is distributed rather than rebuilt

The processed LMDB carries the Vina affinity labels the training objective reads, and those
were written into it by an earlier run. The repository does not contain a script that can put
them back, so the LMDB cannot be regenerated from CrossDocked2020 with what is here.

It also cannot be substituted. `crossdocked_pocket10_pose_split.pt` holds integer indices into
the LMDB's key list, which is in byte-sorted order. A rebuilt LMDB that includes or skips even
one record shifts every index, and training would proceed without complaint on a different
split. So use the archived LMDB, or accept that your split is not ours.

The underlying data are CrossDocked2020, which is public and CC0:
<https://bits.csb.pitt.edu/files/crossdock2020/>. The split is the one introduced by
TargetDiff and used by the models compared against here.
`scripts/data_preparation/clean_crossdocked.py` records the selection rule that produced the
pose set, which keeps pairs within 1 A RMSD of the crystal pose.

## The chain-remapped receptors

Twelve of the 100 test receptors encode their chain copies only in the segID column.
Biopython's PDB parser, and therefore PoseCheck, does not read segID, so it keys several atoms
to the same identity and silently keeps one of each, measuring against a fraction of the
receptor. Those pockets must be measured against `data/test_set_chainfix/`. The archive ships
it, and it is reproducible from `data/test_set/`:

```bash
python scripts/make_chainfix_receptors.py --test_set data/test_set
```
