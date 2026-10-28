# `data/`

Two small files ship here. The rest comes from the archive.

```bash
python tools/download_archive.py --doi 10.5281/zenodo.23029438
gunzip data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb.gz
```

| File | Size | What |
|---|---|---|
| `crossdocked_pocket10_pose_split.pt` | ships | the split: 99,990 train, 0 validation, 100 test, as integer indices into the LMDB |
| `test_pocket_order.json` | ships | the canonical `ligand_filename` of each test pocket, in pocket order |
| `test_set/`, `test_set_chainfix/` | 48 MB packed | the 100 test receptors with their reference ligands, and the chain-remapped copies |
| `crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb` | 1.07 GB packed | the processed training set |

## Pocket order

Pocket identity keys on the full ligand filename, never on the pocket directory: seven
CrossDocked sites appear twice in the test split under one directory with different reference
ligands. Every measurement step reads `test_pocket_order.json` so the exports, the docking
outputs and the interaction profiles agree on which pocket is pocket *i*. It is the split's
own order, and you can re-derive it:

```python
import json, torch
from gated_energy_diffusion.datasets.pl_pair_dataset import PocketLigandPairDataset

ds = PocketLigandPairDataset('data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb')
split = torch.load('data/crossdocked_pocket10_pose_split.pt')
assert [ds.get_ori_data(int(i))['ligand_filename'] for i in split['test']] == \
       json.load(open('data/test_pocket_order.json'))
```

## The training LMDB

Decompressed it reports 4.07 GB and holds about 1.45 GB: LMDB preallocates its map and leaves
the rest a hole, so `ls` and `du --apparent-size` disagree with plain `du`.

It carries the Vina affinity labels the objective reads, and the split indexes its byte-sorted
key order, so a rebuilt LMDB with a different record set shifts every index without raising an
error. Use this file.

The underlying data are CrossDocked2020, CC0:
<https://bits.csb.pitt.edu/files/crossdock2020/>. The split is TargetDiff's.
`scripts/data_preparation/clean_crossdocked.py` records the selection rule, pairs within
1 A RMSD of the crystal pose.

## Chain-remapped receptors

Twelve of the 100 receptors encode their chain copies only in the segID column, which
Biopython and therefore PoseCheck ignores, leaving a partial receptor. Measure against
`test_set_chainfix/`. The archive ships it; to regenerate:

```bash
python scripts/make_chainfix_receptors.py --test_set data/test_set
```
