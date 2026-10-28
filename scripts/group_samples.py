#!/usr/bin/env python3
"""Raw sampler output -> one grouped .pt, which is what every measurement step reads.

    python scripts/group_samples.py --base results/samples --out results/mymodel.pt

bin/sample.py writes one file per molecule, `<base>/id<data_id>/result_<i>.pt`, holding the
predicted coordinates and atom-type indices and nothing else. This rebuilds the RDKit
molecule from those, computes the same `chem_results` the published .pt files carry, and
writes the grouped list the rest of the pipeline expects:

    [ [ {mol, smiles, ligand_filename, chem_results}, ... ],   # pocket 0
      ... ]                                                     # index == pocket order

Reconstruction follows the sampler's own convention: coordinates are cast to float64
(OpenBabel's SetVector rejects float32), atom types are decoded with the `add_aromatic`
encoding, and per-atom affinity weights are ones, which is what the released model stores.

Docking scores are NOT added here. They come from scripts/vina_meeko_dock.py and
scripts/dock_engines.py, which run on the exported SDFs further down the pipeline.

A molecule that cannot be reconstructed keeps its place in the list with `mol` and `smiles`
set to None, which is the convention the published .pt files follow. It is deliberate rather
than tidy: reconstruction fails on distorted geometry, so dropping those entries would
shrink a pocket's denominator by an amount that depends on the model being measured, and
would renumber every molecule after them. The downstream steps skip null entries and report
how many they skipped.
"""
import argparse
import os
import re
import sys
from glob import glob
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, RDLogger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gated_energy_diffusion.utils import reconstruct, transforms            # noqa: E402
from gated_energy_diffusion.utils.evaluation.scoring_func import get_chem   # noqa: E402

RDLogger.DisableLog('rdApp.*')


def canonical_ligand_files(path):
    import json
    with open(path) as fh:
        return json.load(fh)


def rebuild_mol(raw, atom_enc_mode='add_aromatic'):
    """raw {'pos','v'} -> mol, or None if reconstruction fails."""
    if raw is None or 'pos' not in raw or 'v' not in raw:
        return None
    pos = np.asarray(raw['pos'], dtype=np.float64)
    v = np.asarray(raw['v'])
    try:
        z = transforms.get_atomic_number_from_index(v, mode=atom_enc_mode)
        aromatic = transforms.is_aromatic_from_index(v, mode=atom_enc_mode)
        mol = reconstruct.reconstruct_from_generated(pos, z, aromatic, np.ones_like(v))
    except Exception:                                        # noqa: BLE001
        return None
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    return mol


def chem_results_of(mol):
    """get_chem on a sanitised copy, or None when it cannot be computed."""
    m = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(m)
        return get_chem(m)
    except Exception:                                        # noqa: BLE001
        return None


def pocket_dirs(base):
    """{pocket_idx: path} for <base>/id<N> and <base>/id<N>_<tag>."""
    out = {}
    for d in sorted(glob(os.path.join(base, 'id*'))):
        m = re.match(r'id(\d+)', os.path.basename(d))
        if m and os.path.isdir(d):
            out.setdefault(int(m.group(1)), d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True, help="the sampler's --result_path (holds id<N>/)")
    ap.add_argument('--out', required=True, help='output .pt path')
    ap.add_argument('--pocket_order', default=str(ROOT / 'data' / 'test_pocket_order.json'))
    ap.add_argument('--atom_enc_mode', default='add_aromatic')
    a = ap.parse_args()

    canon = canonical_ligand_files(a.pocket_order)
    dirs = pocket_dirs(a.base)
    if not dirs:
        sys.exit(f'no id<N>/ directories under {a.base}')
    unknown = [i for i in dirs if i >= len(canon)]
    if unknown:
        sys.exit(f'pocket index outside the {len(canon)}-pocket order: {sorted(unknown)[:5]}')

    grouped = [[] for _ in canon]
    n_mol = n_bad_mol = n_bad_chem = 0
    for pi in sorted(dirs):
        files = sorted(glob(os.path.join(dirs[pi], 'result_*.pt')),
                       key=lambda p: int(re.search(r'result_(\d+)', p).group(1)))
        for f in files:
            n_mol += 1
            mol = rebuild_mol(torch.load(f, map_location='cpu'), a.atom_enc_mode)
            if mol is None:
                n_bad_mol += 1
                grouped[pi].append({'mol': None, 'smiles': None,
                                    'ligand_filename': canon[pi], 'chem_results': None})
                continue
            chem = chem_results_of(mol)
            if chem is None:
                n_bad_chem += 1
            grouped[pi].append({'mol': mol, 'smiles': Chem.MolToSmiles(mol),
                                'ligand_filename': canon[pi], 'chem_results': chem})
        built = sum(1 for e in grouped[pi] if e['mol'] is not None)
        print(f'pocket {pi:3d}: {len(files):4d} files -> {built:4d} molecules '
              f'({len(grouped[pi]) - built} not reconstructable)', flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or '.', exist_ok=True)
    torch.save(grouped, a.out)
    kept = sum(1 for g in grouped for e in g if e["mol"] is not None)
    print(f'\n{len(dirs)} pockets, {n_mol} sampled, {kept} reconstructed, '
          f'{n_bad_mol} not reconstructable, {n_bad_chem} without chem_results -> {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
