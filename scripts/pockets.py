#!/usr/bin/env python3
"""Canonical CrossDocked test-pocket order, from the sha256-gated single source of truth.

Four code paths derive this order today:
  eval_export_sdf.canonical_ligand_files    reads targetdiff_vina_docked.pt
  plip_interactions.canonical_ligand_files  a second copy of the same
  build_comparison_tables.canonical_index + build_pocket_maps
  delta_score_common.load_test_pockets      already reads test_pockets.json (the SSOT)

They AGREE -- `scripts/check_pockets_equivalence.py` measures it rather than trusting it -- so
this module is a consolidation, not a correction. It reads `analysis/nci_benchmark/test_pockets.json`
directly, which is both cheaper (no torch, no .pt load) and the declared canonical source.

    from pockets import canonical
    idx2full, full2idx, dir2idx, idx2name = canonical()

THE DIRECTORY IS NOT A KEY. Seven CrossDocked test directories host TWO binding sites each under
different reference ligands (NOS1_HUMAN at indices 53 and 54, and pairs 14/15, 20/21, 56/57,
65/66, 87/88, 91/92). Keying on the directory merges two distinct targets: it once inflated a
850-molecule set to 966, and a related renumbering bug paired 77 PIDiff pockets with the wrong
receptor. `dir2idx` therefore contains ONLY directories unique to a single index, and everything
else must go through `full2idx` on the complete ligand_filename.
"""
import hashlib
import json
import os
from collections import Counter

SSOT = 'analysis/nci_benchmark/test_pockets.json'

_CACHE = {}


def load(path=SSOT):
    key = os.path.abspath(path)
    if key not in _CACHE:
        with open(path, encoding='utf-8') as fh:
            _CACHE[key] = json.load(fh)
    return _CACHE[key]


def ligand_files(path=SSOT):
    """Ordered ligand_filename per data_id -- the `canonical_ligand_files()` shape."""
    return [p['ligand_filename'] for p in load(path)['pockets']]


def compute_sha256(path=SSOT):
    """sha256 of the ordered ligand_filename list, as scripts/build_test_pockets.py defines it."""
    return hashlib.sha256('\n'.join(ligand_files(path)).encode()).hexdigest()


def verify_sha256(path=SSOT, assignment=None):
    """Raise unless the SSOT still hashes to its recorded value (and the assignment agrees).

    This is the assertion the three legacy code paths never had. The off-target assignment
    stores the same hash precisely so a silently edited pocket list cannot go unnoticed.
    """
    tp = load(path)
    want, got = tp.get('sha256_ligand_filenames'), compute_sha256(path)
    if want != got:
        raise RuntimeError(f'{path}: ligand_filename list has changed -- '
                           f'recorded {want}, computed {got}')
    if assignment and os.path.isfile(assignment):
        with open(assignment, encoding='utf-8') as fh:
            asg = json.load(fh)
        if asg.get('test_pockets_sha256') != want:
            raise RuntimeError(f'{assignment}: test_pockets_sha256 '
                               f'{asg.get("test_pockets_sha256")} != {want}')
    return want


def canonical(path=SSOT):
    """-> (idx2full, full2idx, dir2idx, idx2name).

    idx2full  pocket index -> full ligand_filename
    full2idx  full ligand_filename -> index (exact, 1:1)
    dir2idx   pocket dir -> index, ONLY for dirs unique to one index (see the module docstring)
    idx2name  display name per index; a shared dir is tagged with its receptor code so the two
              rows stay visually distinct

    Body transplanted from build_comparison_tables.build_pocket_maps so the maps are identical.
    """
    verify_sha256(path)
    idx2full = {p['pocket_idx']: p['ligand_filename'] for p in load(path)['pockets']}
    dir_of = {i: lf.split('/')[0] for i, lf in idx2full.items()}
    dir_counts = Counter(dir_of.values())
    idx2name, full2idx, dir2idx = {}, {}, {}
    for i, lf in idx2full.items():
        d = dir_of[i]
        full2idx[lf] = i
        if dir_counts[d] == 1:
            dir2idx[d] = i
            idx2name[i] = d
        else:                                    # shared dir -> tag by receptor
            ref = os.path.basename(lf).split('_')[0]
            idx2name[i] = f'{d} [{ref}]'
    return idx2full, full2idx, dir2idx, idx2name


def shared_dirs(path=SSOT):
    """{dir: [indices]} for the directories hosting more than one binding site."""
    idx2full, _, _, _ = canonical(path)
    by_dir = {}
    for i, lf in idx2full.items():
        by_dir.setdefault(lf.split('/')[0], []).append(i)
    return {d: sorted(v) for d, v in by_dir.items() if len(v) > 1}


if __name__ == '__main__':
    print(f'{SSOT}: sha256 {verify_sha256()} OK')
    idx2full, full2idx, dir2idx, idx2name = canonical()
    print(f'  {len(idx2full)} pockets, {len(full2idx)} exact keys, '
          f'{len(dir2idx)} directory keys ({len(idx2full) - len(dir2idx)} need the full path)')
    print('  directories hosting two binding sites:')
    for d, idxs in sorted(shared_dirs().items()):
        print(f'    {d:24s} {idxs}  ->  {[idx2name[i] for i in idxs]}')
