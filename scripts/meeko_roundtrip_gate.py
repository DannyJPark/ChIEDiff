#!/usr/bin/env python3
"""Gate for the meeko/Vina RMSD track: prove the PDBQT round-trip preserves the molecule.

Phase 1B computes RMSD(generated, Vina-docked) by preparing the generated mol with meeko,
docking it, and rebuilding an RDKit mol from the output PDBQT. That is only valid if
prepare -> write_pdbqt -> read_back is an IDENTITY on the molecule. This script checks it
BEFORE ~57k dockings get queued.

What "identity" does and does NOT mean here
-------------------------------------------
meeko rebuilds the mol in ITS OWN canonical-SMILES atom order, not the input SDF order --
the `REMARK SMILES IDX` lines in the PDBQT are that mapping. So comparing atom i to atom i
is meaningless and will report ~5 A of nonsense. The correct check, and the metric Phase 1B
actually uses, is rdMolAlign.CalcRMS: no superposition, symmetry-aware, matches by graph.
A clean round-trip gives CalcRMS ~= 5e-4 A, which is PDBQT's 3-decimal coordinate precision,
not a real displacement.

Fragmented molecules
--------------------
meeko refuses a disconnected mol ("RDKit molecule has N fragments. Must have 1."). That is a
property of the GENERATED molecules, not a bug -- PoseBusters already reports all_atoms_connected
failing at double-digit rates. Those molecules get prep_ok=0 rather than being dropped silently,
because the fragmentation rate DIFFERS BY MODEL and would otherwise re-introduce exactly the
MNAR coverage bias (D3) this whole experiment exists to remove.

Usage:
    ~/anaconda3/envs/meekovina/bin/python scripts/meeko_roundtrip_gate.py --n 60
"""
import argparse
import json
import os
import sys

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolAlign

from meeko import MoleculePreparation, PDBQTMolecule, PDBQTWriterLegacy, RDKitMolCreate

RDLogger.DisableLog('rdApp.*')

# PDBQT stores coordinates to 3 decimals, so a perfect round-trip still shows ~1e-3 A of
# quantisation. Anything above this is a real change to the molecule.
RMSD_TOL = 5e-3


def prepare_pdbqt(mol):
    """RDKit mol (with Hs, 3D) -> PDBQT string. Returns (pdbqt, None) or (None, reason)."""
    try:
        setups = MoleculePreparation().prepare(mol)
    except ValueError as e:                                   # noqa: PERF203
        msg = str(e)
        return None, 'fragmented' if 'fragment' in msg else f'prep_error:{msg[:40]}'
    except Exception as e:                                    # noqa: BLE001
        return None, f'prep_exc:{type(e).__name__}'
    if not setups:
        return None, 'no_setup'
    s, ok, err = PDBQTWriterLegacy.write_string(setups[0])
    if not ok:
        return None, f'write_failed:{str(err)[:40]}'
    return s, None


def heavy(mol):
    """Heavy atoms only, never raising on a bad valence."""
    try:
        mol = Chem.RemoveHs(mol, sanitize=False)
    except Exception:                                         # noqa: BLE001
        pass
    if any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()):
        rw = Chem.RWMol(mol)
        for i in sorted((a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 1), reverse=True):
            rw.RemoveAtom(i)
        mol = rw.GetMol()
    return mol


def check_one(mol):
    """-> (status, rmsd_or_None). status in {'ok','fragmented',...}"""
    mh = Chem.AddHs(mol, addCoords=True)
    pdbqt, reason = prepare_pdbqt(mh)
    if pdbqt is None:
        return reason, None
    try:
        back = RDKitMolCreate.from_pdbqt_mol(PDBQTMolecule(pdbqt, skip_typing=True))[0]
    except Exception as e:                                    # noqa: BLE001
        return f'readback_exc:{type(e).__name__}', None
    if back is None:
        return 'readback_none', None
    a, b = heavy(mol), heavy(back)
    if a.GetNumAtoms() != b.GetNumAtoms():
        return 'atom_count_changed', None
    try:
        # CalcRMS: no superposition + symmetry-aware. NOT GetBestRMS (that would align).
        return 'ok', float(rdMolAlign.CalcRMS(b, a))
    except Exception as e:                                    # noqa: BLE001
        return f'calcrms_exc:{type(e).__name__}', None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--registry', default='configs/models.json')
    ap.add_argument('--n', type=int, default=60, help='molecules sampled per model')
    ap.add_argument('--pockets', type=int, default=6, help='pockets sampled per model')
    ap.add_argument('--out', default='results/pose_fidelity/meeko_roundtrip_gate.json')
    args = ap.parse_args()

    import sys as _sys, os as _os

    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

    import model_registry                                    # noqa: E402

    reg = model_registry.load_as(args.registry, 'pose')
    rng = np.random.default_rng(0)
    report, gate_ok = {}, True

    print(f'{"model":14s} {"n":>5s} {"ok":>5s} {"frag":>5s} {"other":>6s} '
          f'{"maxRMSD":>9s} {"verdict":>8s}')
    print('-' * 62)
    for m in reg['models']:
        files = sorted(f for f in os.listdir(os.path.join(m['dir'], 'sdf')) if f.endswith('.sdf'))
        pick = rng.choice(len(files), size=min(args.pockets, len(files)), replace=False)
        stats, rmsds = {}, []
        n_seen = 0
        for pi in pick:
            path = os.path.join(m['dir'], 'sdf', files[pi])
            mols = [x for x in Chem.SDMolSupplier(path, removeHs=False) if x is not None]
            for mol in mols[:max(1, args.n // len(pick))]:
                st, r = check_one(mol)
                key = st if st in ('ok', 'fragmented') else 'other'
                stats[key] = stats.get(key, 0) + 1
                if st != 'ok':
                    stats.setdefault('_detail', {})
                    stats['_detail'][st] = stats['_detail'].get(st, 0) + 1
                else:
                    rmsds.append(r)
                n_seen += 1
        r = np.array(rmsds) if rmsds else np.array([np.nan])
        n_ok, n_frag = stats.get('ok', 0), stats.get('fragmented', 0)
        n_other = stats.get('other', 0)
        # gate: every molecule meeko ACCEPTS must round-trip exactly. Fragmented ones are a
        # property of the model, tracked separately, and must not fail the gate.
        ok = n_ok > 0 and np.nanmax(r) < RMSD_TOL and n_other == 0
        gate_ok &= ok
        print(f'{m["tag"]:14s} {n_seen:5d} {n_ok:5d} {n_frag:5d} {n_other:6d} '
              f'{np.nanmax(r):9.6f} {"PASS" if ok else "FAIL":>8s}')
        report[m['tag']] = {
            'n_sampled': n_seen, 'n_ok': n_ok, 'n_fragmented': n_frag, 'n_other': n_other,
            'frag_rate': round(n_frag / n_seen, 4) if n_seen else None,
            'max_rmsd': None if np.isnan(np.nanmax(r)) else round(float(np.nanmax(r)), 8),
            'median_rmsd': None if np.isnan(np.nanmax(r)) else round(float(np.nanmedian(r)), 8),
            'detail': stats.get('_detail', {}),
        }

    print('-' * 62)
    print(f'ROUND-TRIP GATE: {"PASS" if gate_ok else "FAIL"}   (tolerance {RMSD_TOL} A)')
    print('\nfragmentation rate per model (meeko cannot prepare these; tracked, not dropped):')
    for t, d in sorted(report.items(), key=lambda kv: -(kv[1]['frag_rate'] or 0)):
        print(f'  {t:14s} {100 * (d["frag_rate"] or 0):5.1f}%   ({d["n_fragmented"]}/{d["n_sampled"]})')
    print('\nNOTE: the rate differs by model, so restricting the meeko/Vina arm to connected')
    print('      molecules is NOT a neutral filter. The smina/SDF arm has no such restriction')
    print('      and is the coverage control -- compare rankings across the two.')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'tolerance': RMSD_TOL, 'gate_pass': bool(gate_ok), 'models': report}, f, indent=2)
    print(f'\n-> {args.out}')
    return 0 if gate_ok else 1


if __name__ == '__main__':
    sys.exit(main())
