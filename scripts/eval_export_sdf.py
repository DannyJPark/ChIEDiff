"""Export generated molecules from a sampling .pt into per-pocket SDF files.

The .pt layout is the grouped-by-pocket one used across this repo:
    list[N_pockets] of list[mol_dict], each dict holding at least
    {'mol': rdkit.Mol (3D), 'smiles': str, 'ligand_filename': '<POCKET>/<rec>_..._lig_....sdf'}

Writes:
    <out>/sdf/pocket<NNN>.sdf   one multi-model SDF per pocket, mols named p<NNN>_m<MMMM>
    <out>/manifest.csv          pocket_idx, pocket, receptor, n_mols, ref_ligand

The mol name p<NNN>_m<MMMM> carries the pocket index and the ORIGINAL index inside that pocket's
list, so gnina / smina / PoseBusters rows can be joined straight back onto the source .pt even
though mols that fail to load are skipped. All three tools preserve the SDF title line.

ref_ligand is the crystal ligand for that pocket; the docking modes use it as the box
(--autobox_ligand), which is what "redock into the known pocket" means here.

Usage:
    python scripts/eval_export_sdf.py --pt results/sampling_results/targetdiff_vina_docked.pt \
        --out eval_gnina/targetdiff
"""
import argparse
import csv
import os

import torch
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')


def paths_for(ligand_filename, test_set_root):
    """'<POCKET>/<pdb>_<ch>_rec_<...>_lig_tt_docked_3.sdf'
       -> (pocket, '<test_set>/<POCKET>/<pdb>_<ch>_rec.pdb', '<test_set>/<ligand_filename>')"""
    pocket = os.path.dirname(ligand_filename)
    stem = os.path.basename(ligand_filename).split('_rec')[0] + '_rec.pdb'
    receptor = os.path.join(test_set_root, pocket, stem)
    ref_ligand = os.path.join(test_set_root, ligand_filename)
    return pocket, receptor, ref_ligand


def _torch_load(path):
    """Load a sampling .pt under either torch generation.

    These files were written in the kgdiff env (torch 1.11), whose `torch.load` has no
    `weights_only` kwarg. torch >= 2.6 defaults it to True and then refuses to unpickle the
    `rdkit.Chem.rdchem.Mol` / `numpy.core.multiarray.scalar` globals inside them, raising
    UnpicklingError -- so the ordering must be try-new-kwarg-first, never the reverse.
    """
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:                                        # torch < 1.13: no such kwarg
        return torch.load(path, map_location='cpu')


def load_grouped(path):
    """Normalise the three .pt layouts in this repo to list[pocket] of list[mol_dict].

    The sampling sets are not stored uniformly:
      * grouped     : list[N_pockets] of list[mol]        (targetdiff, kgdiff, consolidate_docking)
      * dict-wrapped: {'all_results': [mol, ...], ...}     (PIDiff) -- a FLAT list, one entry per
                      molecule, so it has to be regrouped by ligand_filename
      * flat        : [mol, ...]                           (same, without the dict wrapper)

    Pocket ORDER matters: index i must be the CrossDock test-set data_id, because every downstream
    join (manifest -> receptor, p<NNN>_m<MMMM> titles, cross-model per-pocket tables) keys on it.

    Flat layouts must therefore be placed at their CANONICAL index, not renumbered by order of
    appearance. PIDiff generated nothing for one pocket; renumbering by appearance packed the rest
    into 0..91 and silently shifted 77 of them onto the WRONG receptor. So flat sets are aligned
    through the canonical order (targetdiff's), and pockets the model has no molecules for stay
    present as empty slots.

    Alignment keys on the FULL ligand_filename, not the pocket directory: the CrossDock test set
    lists seven binding sites twice under the same directory with a different reference ligand
    (e.g. NOS1 at data_id 53 and 54), so a directory-keyed map would copy one pocket's molecules
    into both slots and inflate the molecule count. Same reasoning as build_comparison_tables.py's
    full2idx. A molecule whose exact ligand_filename is not in the canonical set falls back to the
    directory, but only when that directory maps to exactly one index.
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


def canonical_ligand_files(path='results/sampling_results/targetdiff_vina_docked.pt'):
    """Canonical ligand_filename per CrossDock data_id, from the canonical grouped set."""
    if not os.path.isfile(path):
        return []
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
            # not fatal: only the docking modes need the box
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

    # lineterminator='\n': the csv 'excel' dialect defaults to '\r\n', and with newline='' Python
    # writes that CR literally. manifest.csv is read by the sbatch scripts with `IFS=, read`, which
    # would then carry a trailing '\r' into the LAST field -- ref_ligand -- and hand
    # `--autobox_ligand <path>\r` to smina/gnina, i.e. a file that does not exist.
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
