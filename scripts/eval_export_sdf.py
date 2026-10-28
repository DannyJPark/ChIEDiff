"""Export generated molecules from a sampling .pt into one multi-model SDF per pocket, plus a
manifest.

The mol name p<NNN>_m<MMMM> carries the pocket index and the molecule's own index inside that
pocket's list, so gnina, smina and PoseBusters rows can be joined straight back onto the source
.pt even though mols that fail to load are skipped. All three tools preserve the SDF title line.

ref_ligand is the crystal ligand for that pocket, which the docking modes pass as the box.
"""
import argparse
import csv
import json
import os

import torch
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')


def paths_for(ligand_filename, test_set_root):
    """Map a ligand_filename to (pocket, receptor pdb, reference ligand sdf)."""
    pocket = os.path.dirname(ligand_filename)
    stem = os.path.basename(ligand_filename).split('_rec')[0] + '_rec.pdb'
    receptor = os.path.join(test_set_root, pocket, stem)
    ref_ligand = os.path.join(test_set_root, ligand_filename)
    return pocket, receptor, ref_ligand


def _torch_load(path):
    """Load a sampling .pt under either torch generation.

    torch 2.6 and later default weights_only to True and then refuse to unpickle the
    rdkit.Chem.rdchem.Mol and numpy.core.multiarray.scalar globals these files contain, while
    torch 1.11 has no such kwarg. The kwarg therefore has to be tried first, never the other
    way round.
    """
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:                                        # torch < 1.13: no such kwarg
        return torch.load(path, map_location='cpu')


def load_grouped(path):
    """Normalise a grouped, flat or dict-wrapped .pt to list[pocket] of list[mol_dict].

    Index i must be the CrossDocked test-set data_id, because every downstream join keys on it.
    A flat layout therefore goes to its canonical index rather than being renumbered by order of
    appearance. One model that generated nothing for a single pocket shows the hazard, since
    renumbering by appearance packs the remainder into 0..91 and shifts 77 of them onto the wrong
    receptor. Pockets a model has no molecules for stay present as empty slots.

    Alignment keys on the full ligand_filename and never on the pocket directory. The CrossDocked
    test set lists seven binding sites twice under one directory with a different reference
    ligand, so a directory-keyed map would copy one pocket's molecules into both slots. A molecule
    whose exact ligand_filename is absent falls back to the directory only when that directory
    maps to exactly one index.
    """
    obj = _torch_load(path)
    if isinstance(obj, dict):
        obj = obj.get('all_results', obj.get('results', []))
    if not obj:
        return []
    if isinstance(obj[0], list):
        return obj                                    # already grouped, index == data_id

    canon = canonical_ligand_files()
    if not canon:                                     # no canonical file -> appearance order
        groups, order = {}, []
        for e in obj:
            lf = e.get('ligand_filename')
            if lf is None:
                continue
            groups.setdefault(os.path.dirname(lf), []).append(e)
            if os.path.dirname(lf) not in order:
                order.append(os.path.dirname(lf))
        return [groups[k] for k in order]

    full2idx = {lf: i for i, lf in enumerate(canon) if lf}
    dir_counts = {}
    for lf in canon:
        if lf:
            d = os.path.dirname(lf)
            dir_counts[d] = dir_counts.get(d, 0) + 1
    dir2idx = {os.path.dirname(lf): i for i, lf in enumerate(canon)
               if lf and dir_counts[os.path.dirname(lf)] == 1}

    out = [[] for _ in canon]
    n_unmatched = 0
    for e in obj:
        lf = e.get('ligand_filename')
        if lf is None:
            continue
        i = full2idx.get(lf)
        if i is None:
            i = dir2idx.get(os.path.dirname(lf))      # unique-dir fallback only
        if i is None:
            n_unmatched += 1
            continue
        out[i].append(e)
    if n_unmatched:
        print(f'[warn] {n_unmatched} molecule(s) matched no canonical pocket and were dropped')
    return out


def canonical_ligand_files(path='data/test_pocket_order.json'):
    """The canonical ligand_filename of each test pocket, in pocket order. A .pt of grouped
    results is also accepted and its order is read the same way."""
    if not os.path.isfile(path):
        return []
    if path.endswith('.json'):
        with open(path) as fh:
            return json.load(fh)
    obj = _torch_load(path)
    out = []
    for elem in obj:
        mols = elem if isinstance(elem, list) else [elem]
        lf = None
        for m in mols:
            if isinstance(m, dict) and m.get('ligand_filename'):
                lf = m['ligand_filename']
                break
        out.append(lf)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pt', required=True, help='sampling .pt (grouped, flat, or dict-wrapped)')
    ap.add_argument('--out', required=True, help='output dir')
    ap.add_argument('--test_set', default='data/test_set')
    args = ap.parse_args()

    data = load_grouped(args.pt)
    sdf_dir = os.path.join(args.out, 'sdf')
    os.makedirs(sdf_dir, exist_ok=True)

    rows, n_written, n_skipped, n_bad_pocket = [], 0, 0, 0
    for pi, pocket_mols in enumerate(data):
        if not pocket_mols:
            continue
        pocket, rec, ref = paths_for(pocket_mols[0]['ligand_filename'], args.test_set)
        if not os.path.exists(rec):
            n_bad_pocket += 1
            print(f'[WARN] pocket {pi:03d} ({pocket}): receptor missing -> {rec}; skipping pocket')
            continue
        if not os.path.exists(ref):
            # Not fatal, because only the docking modes need the box.
            print(f'[WARN] pocket {pi:03d} ({pocket}): reference ligand missing -> {ref}; '
                  f'docking modes will skip this pocket')
            ref = ''

        path = os.path.join(sdf_dir, f'pocket{pi:03d}.sdf')
        w = Chem.SDWriter(path)
        n = 0
        for mi, entry in enumerate(pocket_mols):
            mol = entry.get('mol')
            if mol is None or mol.GetNumConformers() == 0:
                n_skipped += 1
                continue
            mol.SetProp('_Name', f'p{pi:03d}_m{mi:04d}')
            w.write(mol)
            n += 1
        w.close()
        n_written += n
        rows.append({'pocket_idx': pi, 'pocket': pocket, 'receptor': rec,
                     'n_mols': n, 'ref_ligand': ref})

    # lineterminator is required, because the csv excel dialect would write a literal CR that
    # shell callers reading with IFS=, read carry into ref_ligand, naming a file that does not
    # exist when it reaches smina or gnina.
    with open(os.path.join(args.out, 'manifest.csv'), 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=['pocket_idx', 'pocket', 'receptor', 'n_mols', 'ref_ligand'],
                            lineterminator='\n')
        wr.writeheader()
        wr.writerows(rows)

    print(f'pockets exported : {len(rows)}')
    print(f'molecules written: {n_written}')
    print(f'mols skipped (no mol / no 3D conformer): {n_skipped}')
    print(f'pockets skipped (receptor missing)     : {n_bad_pocket}')
    print(f'-> {sdf_dir}, manifest at {os.path.join(args.out, "manifest.csv")}')


if __name__ == '__main__':
    main()
