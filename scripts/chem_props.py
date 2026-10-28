#!/usr/bin/env python3
"""Per-molecule chemical properties: QED, SA, LogP, Lipinski, ring sizes.

    python scripts/chem_props.py --dir eval_out/mymodel

The properties come from gated_energy_diffusion.utils.evaluation.scoring_func.get_chem, the
same function whose output the sampler stores under `chem_results` in the .pt files.

This is for measuring a model of your own. It does NOT reproduce the published property
columns exactly, and the difference is in the input rather than the code: the sampler
computes `chem_results` on the RDKit molecule it reconstructed in memory, whereas this
reads the molecule back from an SDF. Writing a molecule with unusual valences to SDF and
reading it again lets RDKit re-perceive the valence, which moves implicit hydrogens into
radical electrons. Measured on one model, three pockets, 282 molecules: ring-size counts
agree for every molecule, and QED, SA and LogP differ for ten of them (3.5%); one such
molecule lost two hydrogens to two radical electrons, 520.319 -> 518.303 Da. So the
published values are the ones in `chem_results`, which reproduction.ipynb reads directly
from the .pt, and these are the ones to use when the SDFs are all you have.

Molecules are read with sanitize=False and sanitized afterwards, one at a time. A molecule
RDKit cannot sanitize gets a row with empty properties and `status` saying so, rather than
disappearing: sanitization fails on distorted geometry, which is not independent of the
models being compared, so dropping those molecules would bias every column that follows.

Input:  <dir>/sdf/pocket<NNN>.sdf, from scripts/eval_export_sdf.py
Output: <dir>/chem_props.csv
"""
import argparse
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rdkit import Chem, RDLogger                                      # noqa: E402

from gated_energy_diffusion.utils.evaluation.scoring_func import get_chem   # noqa: E402

RDLogger.DisableLog('rdApp.*')

RINGS = range(3, 10)
FIELDS = (['name', 'pocket_idx', 'heavy', 'n_frags', 'qed', 'sa', 'logp', 'lipinski']
          + [f'ring{n}' for n in RINGS] + ['ring_other', 'status'])


def one_molecule(mol, pocket_idx, name):
    rec = {'name': name, 'pocket_idx': pocket_idx,
           'heavy': mol.GetNumHeavyAtoms(), 'status': 'ok'}
    try:
        rec['n_frags'] = len(Chem.GetMolFrags(mol))
    except Exception:                                        # noqa: BLE001
        rec['n_frags'] = ''
    m = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(m)
    except Exception as e:                                   # noqa: BLE001
        rec['status'] = f'sanitize_failed: {type(e).__name__}'
        return rec
    try:
        chem = get_chem(m)
    except Exception as e:                                   # noqa: BLE001
        rec['status'] = f'get_chem_failed: {type(e).__name__}'
        return rec
    rec['qed'] = round(float(chem['qed']), 5)
    rec['sa'] = round(float(chem['sa']), 5)
    rec['logp'] = round(float(chem['logp']), 5)
    rec['lipinski'] = int(chem['lipinski'])
    sizes = chem['ring_size']
    for n in RINGS:
        rec[f'ring{n}'] = int(sizes.get(n, 0))
    rec['ring_other'] = int(sum(v for k, v in sizes.items() if k not in RINGS))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True, help='eval_out/<model> (holds sdf/)')
    ap.add_argument('--pockets', type=int, nargs='+', help='restrict to these pocket_idx')
    a = ap.parse_args()

    sdf_dir = Path(a.dir) / 'sdf'
    files = sorted(f for f in os.listdir(sdf_dir)
                   if f.startswith('pocket') and f.endswith('.sdf'))
    rows = []
    for f in files:
        pi = int(f[6:9])
        if a.pockets and pi not in a.pockets:
            continue
        for i, mol in enumerate(Chem.SDMolSupplier(str(sdf_dir / f), sanitize=False,
                                                   removeHs=False)):
            if mol is None:                   # unreadable record, counted in the printed total
                rows.append({'name': f'p{pi:03d}_m{i:04d}', 'pocket_idx': pi,
                             'status': 'unreadable'})
                continue
            name = mol.GetProp('_Name') if mol.HasProp('_Name') else f'p{pi:03d}_m{i:04d}'
            rows.append(one_molecule(mol, pi, name))

    out = Path(a.dir) / 'chem_props.csv'
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, restval='', lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    ok = sum(1 for r in rows if r['status'] == 'ok')
    print(f'{len(rows)} molecules, {ok} with properties, {len(rows) - ok} without -> {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
