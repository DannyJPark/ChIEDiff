#!/usr/bin/env python3
"""AutoDock Vina score_only + minimize + dock for one (model, pocket), with EXACT atom mapping.

Why this exists alongside the smina/SDF arm
-------------------------------------------
The Vina numbers in the paper come from AutoDock Vina, so the RMSD that accompanies them should
too. The poses already stored in the .pt cannot be reused: they were prepared through the obabel
`PrepLig` path (utils/evaluation/docking_vina.py), which keeps no RDKit index map, so the
correspondence is unrecoverable after the fact. Hence re-docking with meeko, whose PDBQT carries
`REMARK SMILES IDX`.

Meeko returns the rebuilt molecule in CANONICAL SMILES order, not the input SDF order, so all
comparisons go through rmsd_core.rmsd_sym (CalcRMS) and never through atom indices.

Two box modes, both needed:
    ref      crystal reference ligand centroid, fixed cube  -- the protocol all models share
    genbbox  the generated molecule's own bbox              -- reproduces docking_vina.py:214,221,
             i.e. the legacy box, so the legacy pipeline's bias can be isolated with the box held
             fixed, and so T2's required sensitivity analysis exists

Molecules meeko cannot prepare are written with prep_ok=0 and a reason, NEVER dropped: the
fragmentation rate differs by model (15-28% for the physics models, 0% for every baseline), so
silent skipping would rebuild exactly the missing-not-at-random bias this experiment exists to
remove. See results/pose_fidelity/fragmentation_census.csv.

Run with the meekovina env (meeko 0.6.1 + vina 1.2.2):
    ~/anaconda3/envs/meekovina/bin/python scripts/vina_meeko_dock.py \
        --dir eval_out/vina_fixed --pocket 0 --box ref
"""
import argparse
import csv
import os
import sys
import tempfile

import numpy as np
from rdkit import Chem, RDLogger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rmsd_core import compare, coords, heavy                    # noqa: E402

from meeko import MoleculePreparation, PDBQTMolecule, PDBQTWriterLegacy, RDKitMolCreate  # noqa: E402
from vina import Vina                                            # noqa: E402

RDLogger.DisableLog('rdApp.*')

FIELDS = ['name', 'pocket_idx', 'box_mode', 'heavy', 'prep_ok', 'prep_reason',
          'vina_score_only', 'vina_minimize', 'vina_dock',
          'dLocal', 'dGlobal', 'dTotal',
          'rmsd_gen_min', 'rmsd_gen_dock', 'centroid_disp_min', 'centroid_disp_dock',
          'jensen_ok_min', 'jensen_ok_dock', 'charge_fallback']


def prepare_pdbqt(mol_h):
    """-> (pdbqt_string, reason_or_None, used_zero_charge)."""
    for charge_model, is_fallback in ((None, False), ('zero', True)):
        try:
            kw = {} if charge_model is None else {'charge_model': charge_model}
            setups = MoleculePreparation(**kw).prepare(mol_h)
        except ValueError as e:
            msg = str(e)
            # disconnected ligand -- not recoverable, and a real property of the molecule
            return None, ('fragmented' if 'fragment' in msg else f'prep_error:{msg[:60]}'), False
        except Exception as e:                                   # noqa: BLE001
            return None, f'prep_exc:{type(e).__name__}', False
        if not setups:
            return None, 'no_setup', False
        s, ok, err = PDBQTWriterLegacy.write_string(setups[0])
        if ok:
            return s, None, is_fallback
        # Gasteiger can emit NaN on pathological valences (~1%). Vina's scoring function is
        # atom-type based and ignores PDBQT partial charges, so zeroing them is safe.
        if 'non finite charge' not in str(err):
            return None, f'write_failed:{str(err)[:60]}', False
    return None, 'write_failed:nonfinite_charge_after_zero_fallback', False


def read_back(pdbqt_str):
    try:
        return RDKitMolCreate.from_pdbqt_mol(PDBQTMolecule(pdbqt_str, skip_typing=True))[0]
    except Exception:                                            # noqa: BLE001
        return None


def box_for(mode, gen_mol, ref_centroid, size, buffer_):
    """-> (center, box_size). 'genbbox' reproduces docking_vina.py:214/221 exactly."""
    if mode == 'ref':
        return list(ref_centroid), [size, size, size]
    p = coords(heavy(gen_mol))
    center = ((p.max(0) + p.min(0)) / 2).tolist()
    bs = ((p.max(0) - p.min(0)) + buffer_).tolist()
    return center, bs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True, help='eval_out/<model>')
    ap.add_argument('--pocket', type=int, required=True)
    ap.add_argument('--box', choices=['ref', 'genbbox'], default='ref')
    ap.add_argument('--exhaustiveness', type=int, default=16)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--box_size', type=float, default=20.0, help='ref-mode cube edge (A)')
    ap.add_argument('--buffer', type=float, default=5.0, help='genbbox-mode padding (A)')
    ap.add_argument('--limit', type=int, default=0, help='cap molecules (calibration only)')
    ap.add_argument('--mol_start', type=int, default=0, help='first molecule index (chunk parallelism)')
    ap.add_argument('--mol_count', type=int, default=0, help='number of molecules from --mol_start (0=all)')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    tag = f'{args.pocket:03d}'
    sdf = os.path.join(args.dir, 'sdf', f'pocket{tag}.sdf')
    out = args.out or os.path.join(args.dir, 'vina_meeko', f'pocket{tag}_{args.box}.csv')
    if not os.path.isfile(sdf):
        print(f'[skip] no {sdf}')
        return 0

    # manifest: pocket_idx,pocket,receptor,n_mols,ref_ligand
    rec = ref_lig = None
    with open(os.path.join(args.dir, 'manifest.csv')) as f:
        for r in csv.DictReader(f):
            if int(r['pocket_idx']) == args.pocket:
                rec, ref_lig = r['receptor'].strip(), r['ref_ligand'].strip()
                break
    if rec is None:
        print(f'[skip] pocket {args.pocket} not in manifest')
        return 0

    receptor_pdbqt = rec[:-4] + '.pdbqt'
    if not os.path.isfile(receptor_pdbqt):
        print(f'[FAIL] receptor pdbqt missing: {receptor_pdbqt}')
        return 2

    ref_centroid = None
    if args.box == 'ref':
        if not (ref_lig and os.path.isfile(ref_lig)):
            print(f'[skip] pocket {args.pocket}: no reference ligand for ref box')
            return 0
        rm = next(iter(Chem.SDMolSupplier(ref_lig, sanitize=False, removeHs=False)), None)
        if rm is None:
            print(f'[skip] pocket {args.pocket}: unreadable reference ligand')
            return 0
        ref_centroid = coords(heavy(rm)).mean(0)

    mols = [m for m in Chem.SDMolSupplier(sdf, removeHs=False) if m is not None]
    if args.limit:
        mols = mols[:args.limit]
    # chunk parallelism: dock only molecules [mol_start : mol_start+mol_count). The molecule name
    # (p<pocket>_m<idx>) is preserved, so chunk CSVs merge back into one pocket file by name.
    if args.mol_count:
        mols = mols[args.mol_start:args.mol_start + args.mol_count]

    # vina 1.2.2 can only hand back the optimize() pose by writing a file
    tmp_pose = os.path.join(tempfile.gettempdir(),
                            f'vmk_{os.getpid()}_{os.path.basename(args.dir)}_{tag}_{args.box}.pdbqt')
    rows = []
    pose_min, pose_dock = [], []
    for mol in mols:
        name = mol.GetProp('_Name') if mol.HasProp('_Name') else f'p{tag}_m?'
        row = {k: '' for k in FIELDS}
        row.update({'name': name, 'pocket_idx': args.pocket, 'box_mode': args.box,
                    'heavy': heavy(mol).GetNumAtoms(), 'prep_ok': 0, 'charge_fallback': 0})

        mh = Chem.AddHs(mol, addCoords=True)
        pdbqt, reason, zero_fb = prepare_pdbqt(mh)
        if pdbqt is None:
            row['prep_reason'] = reason
            rows.append(row)
            continue
        row['prep_ok'], row['charge_fallback'] = 1, int(zero_fb)

        try:
            center, bsize = box_for(args.box, mol, ref_centroid, args.box_size, args.buffer)
            v = Vina(sf_name='vina', seed=args.seed, verbosity=0)
            v.set_receptor(receptor_pdbqt)
            v.set_ligand_from_string(pdbqt)
            v.compute_vina_maps(center=center, box_size=bsize)

            so = float(v.score()[0])
            # optimize() mutates the internal pose, and its result is only retrievable through
            # write_pose() (there is no .pose() in vina 1.2.2).
            mn = float(v.optimize()[0])
            v.write_pose(tmp_pose, overwrite=True)
            with open(tmp_pose) as fh:
                min_pdbqt = fh.read()
            # Re-seat the ORIGINAL ligand before docking: score_only / minimize / dock must all
            # start from the generated pose, otherwise dock would search from the minimised one
            # and dGlobal would no longer mean what the decomposition says it means.
            v.set_ligand_from_string(pdbqt)
            v.dock(exhaustiveness=args.exhaustiveness, n_poses=1)
            dk = float(v.energies(n_poses=1)[0][0])
            dock_pdbqt = v.poses(n_poses=1)
        except Exception as e:                                   # noqa: BLE001
            row['prep_reason'] = f'vina_exc:{type(e).__name__}:{str(e)[:40]}'
            rows.append(row)
            continue

        row.update({'vina_score_only': round(so, 4), 'vina_minimize': round(mn, 4),
                    'vina_dock': round(dk, 4), 'dLocal': round(mn - so, 4),
                    'dGlobal': round(dk - mn, 4), 'dTotal': round(dk - so, 4)})

        for key, pq, bucket in (('min', min_pdbqt, pose_min), ('dock', dock_pdbqt, pose_dock)):
            back = read_back(pq)
            if back is None:
                continue
            # same_atom_order=False: meeko rebuilds in canonical SMILES order, not input order.
            c = compare(mol, back, same_atom_order=False)
            row[f'rmsd_gen_{key}'] = '' if c['rmsd'] is None else round(c['rmsd'], 4)
            row[f'centroid_disp_{key}'] = ('' if c['centroid_disp'] is None
                                           else round(c['centroid_disp'], 4))
            row[f'jensen_ok_{key}'] = '' if c['jensen_ok'] is None else int(c['jensen_ok'])
            back.SetProp('_Name', name)
            bucket.append(back)
        rows.append(row)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + '.tmp'
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, out)                       # atomic: a preempted job must never look complete

    for key, bucket in (('min', pose_min), ('dock', pose_dock)):
        p = out[:-4] + f'_{key}.sdf'
        wt = Chem.SDWriter(p + '.tmp')
        for m in bucket:
            wt.write(m)
        wt.close()
        os.replace(p + '.tmp', p)

    if os.path.exists(tmp_pose):
        os.unlink(tmp_pose)

    ok = sum(r['prep_ok'] for r in rows)
    frag = sum(1 for r in rows if r['prep_reason'] == 'fragmented')
    bad_j = sum(1 for r in rows if r['jensen_ok_dock'] == 0)
    print(f'[{os.path.basename(args.dir)} p{tag} {args.box}] {len(rows)} mols  '
          f'prep_ok={ok}  fragmented={frag}  jensen_viol={bad_j} -> {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
