#!/usr/bin/env python3
"""Ligand internal strain (MMFF94s) + the size/flexibility confounders, per generated molecule.

Why this metric carries the argument
------------------------------------
Every Vina-derived number in this project is suspect as evidence that our poses are "physically
plausible", because the models were trained with a Vina-derived physics loss: winning on Vina is
guaranteed by construction, not discovered (defect D8). Internal strain is computed with MMFF94s
and never touches the protein or any docking score, so it is genuinely independent evidence.

    E_strain = E_FF(pose as generated) - E_FF(same molecule, freely relaxed)

Both terms use the SAME force field and the SAME molecule, so the difference isolates how far the
generated conformer sits above its own nearest local minimum. Units kcal/mol.

Strain grows roughly linearly with molecule size, so `strain_per_heavy` is the comparable column;
n_rot / heavy / tpsa are emitted alongside because the mixed-effects model in Phase 5 has to
adjust for them (defect D5: RMSD and strain both increase with size and flexibility, so a model
that makes small rigid molecules wins on both without being better).

Disconnected molecules are kept and flagged (`n_frags`), not dropped -- they are 15-28% of the
physics models' output and 0% of every baseline's, so silently skipping them would rebuild the
missing-not-at-random bias (see results/pose_fidelity/fragmentation_census.csv).

Usage:
    python scripts/strain_energy.py --tag vina_fixed
    python scripts/strain_energy.py --all --jobs 8
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
    """-> (strain, e_pose, e_relaxed, status). Never raises."""
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


def run_model(model, jobs):
    sdf_dir = os.path.join(model['dir'], 'sdf')
    tasks = []
    for f in sorted(os.listdir(sdf_dir)):
        if f.startswith('pocket') and f.endswith('.sdf'):
            tasks.append((os.path.join(sdf_dir, f), int(f[6:9])))
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
    ap.add_argument('--registry', default='configs/models.json')
    ap.add_argument('--tag', default='')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--jobs', type=int, default=4)
    args = ap.parse_args()

    import sys as _sys, os as _os

    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

    import model_registry                                    # noqa: E402

    reg = model_registry.load_as(args.registry, 'pose')
    models = [m for m in reg['models'] if args.all or m['tag'] == args.tag]
    if not models:
        raise SystemExit('nothing selected: pass --tag <t> or --all')

    print(f'{"model":14s} {"n":>8s} {"mmff-typable":>21s} {"frag":>10s}  strain/heavy (median)')
    summary = [run_model(m, args.jobs) for m in models]

    print('\nNote: strain is NOT comparable across models without the per-heavy normalisation or')
    print('the Phase 5 size/flexibility adjustment -- it grows with molecule size.')
    print('`strain_status != ok` means MMFF could not type the molecule; that rate is itself a')
    print('validity signal and is reported rather than hidden.')
    return 0


if __name__ == '__main__':
    mp.set_start_method('fork', force=True)
    sys.exit(main())
