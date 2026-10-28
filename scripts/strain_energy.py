#!/usr/bin/env python3
"""Ligand internal strain (MMFF94s) and the size and flexibility covariates, per molecule.

    E_strain = E_FF(pose as generated) - E_FF(same molecule, freely relaxed)

Both terms use the same force field and the same molecule, so the difference isolates how far
the generated conformer sits above its own nearest local minimum, in kcal/mol. Nothing here
touches the protein or a docking score, which is what makes strain independent of the
Vina-derived training signal.

Strain grows roughly linearly with molecule size, so strain_per_heavy is the comparable column.
heavy, n_rot and tpsa are emitted alongside it because strain and pose error both increase with
size and flexibility, and a comparison that ignores them credits a model for making small rigid
molecules.

Disconnected molecules are kept and flagged through n_frags rather than dropped. They are a
large fraction of some models' output and absent from others', so skipping them would make the
missingness depend on the quantity being compared.
"""
import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
from copy import deepcopy

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors

RDLogger.DisableLog('rdApp.*')

FIELDS = ['name', 'pocket_idx', 'heavy', 'n_rot', 'tpsa', 'n_frags',
          'strain_mmff', 'strain_per_heavy', 'e_pose', 'e_relaxed', 'strain_status']


def strain_energy(mol, max_iters=2000):
    """Returns (strain, e_pose, e_relaxed, status) and never raises."""
    try:
        m = Chem.AddHs(deepcopy(mol), addCoords=True)
    except Exception:                                         # noqa: BLE001
        return None, None, None, 'addhs_failed'
    try:
        props = AllChem.MMFFGetMoleculeProperties(m, mmffVariant='MMFF94s')
    except Exception:                                         # noqa: BLE001
        return None, None, None, 'mmff_props_exc'
    if props is None:
        return None, None, None, 'not_mmff_typable'
    try:
        ff0 = AllChem.MMFFGetMoleculeForceField(m, props)
        e_pose = float(ff0.CalcEnergy())
        ff1 = AllChem.MMFFGetMoleculeForceField(m, props)
        ff1.Minimize(maxIts=max_iters)                        # unconstrained relaxation
        e_rel = float(ff1.CalcEnergy())
    except Exception:                                         # noqa: BLE001
        return None, None, None, 'ff_exc'
    if not (np.isfinite(e_pose) and np.isfinite(e_rel)):
        return None, None, None, 'non_finite'
    return e_pose - e_rel, e_pose, e_rel, 'ok'


def one_pocket(task):
    path, pi = task
    rows = []
    for mol in Chem.SDMolSupplier(path, removeHs=False):
        if mol is None:
            continue
        name = mol.GetProp('_Name') if mol.HasProp('_Name') else ''
        try:
            nfrags = len(Chem.GetMolFrags(mol))
        except Exception:                                     # noqa: BLE001
            nfrags = -1
        hv = mol.GetNumHeavyAtoms()
        try:
            nrot = Descriptors.NumRotatableBonds(mol)
            tpsa = round(float(Descriptors.TPSA(mol)), 3)
        except Exception:                                     # noqa: BLE001
            nrot, tpsa = '', ''
        s, ep, er, st = strain_energy(mol)
        rows.append({
            'name': name, 'pocket_idx': pi, 'heavy': hv, 'n_rot': nrot, 'tpsa': tpsa,
            'n_frags': nfrags,
            'strain_mmff': '' if s is None else round(s, 4),
            'strain_per_heavy': '' if (s is None or not hv) else round(s / hv, 5),
            'e_pose': '' if ep is None else round(ep, 4),
            'e_relaxed': '' if er is None else round(er, 4),
            'strain_status': st,
        })
    return rows


def run_model(model, jobs, pockets=None):
    sdf_dir = os.path.join(model['dir'], 'sdf')
    tasks = []
    for f in sorted(os.listdir(sdf_dir)):
        if f.startswith('pocket') and f.endswith('.sdf'):
            pi = int(f[6:9])
            if pockets is None or pi in pockets:
                tasks.append((os.path.join(sdf_dir, f), pi))
    if jobs > 1:
        with mp.Pool(jobs) as pool:
            rows = [r for chunk in pool.imap(one_pocket, tasks) for r in chunk]
    else:
        rows = [r for t in tasks for r in one_pocket(t)]
    rows.sort(key=lambda r: (r['pocket_idx'], r['name']))

    out = os.path.join(model['dir'], 'strain.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)

    vals = np.array([r['strain_per_heavy'] for r in rows
                     if isinstance(r['strain_per_heavy'], float)], dtype=float)
    ok = sum(1 for r in rows if r['strain_status'] == 'ok')
    frag = sum(1 for r in rows if isinstance(r['n_frags'], int) and r['n_frags'] > 1)
    med = float(np.median(vals)) if len(vals) else float('nan')
    print(f'{model["tag"]:14s} n={len(rows):6d} ok={ok:6d} ({100*ok/max(1,len(rows)):5.1f}%) '
          f'frag={frag:5d}  median strain/heavy={med:7.4f} kcal/mol/atom -> {out}')
    return model['tag'], len(rows), ok, frag, med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True, help='eval_out/<model> (holds sdf/)')
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--pockets', type=int, nargs='+', help='restrict to these pocket_idx')
    args = ap.parse_args()

    print(f'{"model":14s} {"n":>8s} {"mmff-typable":>21s} {"frag":>10s}  strain/heavy (median)')
    run_model({'dir': args.dir, 'tag': os.path.basename(args.dir.rstrip("/"))},
              args.jobs, set(args.pockets) if args.pockets else None)

    print('\nStrain is not comparable across models without the per-heavy normalisation or an')
    print('adjustment for size and flexibility: it grows with molecule size.')
    print('`strain_status != ok` means MMFF could not type the molecule. That rate is itself a')
    print('validity signal, so it is reported rather than dropped.')
    return 0


if __name__ == '__main__':
    mp.set_start_method('fork', force=True)
    sys.exit(main())
