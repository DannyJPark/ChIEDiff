#!/usr/bin/env python3
"""
Consolidate the per-pocket docking JSONs produced by dock_generated_ligands.py
(<BASE>/id{N}/docking_results/docking_results.json) into ONE .pt file in the
grouped-by-pocket list format that scripts/build_comparison_tables.py reads:

    obj = [ [mol, mol, ...],   # pocket 0
            [mol, mol, ...],   # pocket 1
            ... ]              # index i == CrossDock test-set data_id i (canonical order)

Each mol dict carries the same keys the reference sets (targetdiff / kgdiff) carry, so every
model can be consumed uniformly -- by build_comparison_tables.py AND by the gnina/smina/
PoseBusters pipeline (eval_export_sdf.py -> run_full_eval.sh), which needs a real molecule:

    {'mol': rdkit.Mol (3D),        <- reconstructed here; see "Molecules" below
     'smiles': str,
     'pred_pos': (N,3) ndarray,    <- generated coordinates, UNTRANSLATED
     'pred_v': (N,) ndarray,       <- generated atom-type indices
     'vina': <vina_results: score_only/minimize/dock>,
     'chem_results': {'qed','sa',...},
     'ligand_filename': <from the canonical map, for robustness>}

Molecules
---------
The docking JSONs record scores but not the molecule, so the mol is rebuilt from the raw sampler
output: every JSON names the `source_file` it came from (e.g. 'result_9.pt'), which is loaded
from <BASE>/id{N}/ and reconstructed with utils.reconstruct.

Reconstruction uses the RAW `pos`, NOT the translated coordinates dock_generated_ligands.py docks
with (it re-centres every ligand onto the reference ligand). targetdiff and kgdiff store `mol`
coordinates exactly equal to `pred_pos` -- verified -- so using the docking frame here would put
this model alone in a different coordinate frame and quietly invalidate any cross-model pose
comparison.

Each rebuilt molecule is cross-checked against the SMILES the docking run already recorded.
Translation preserves connectivity, so the two must agree; disagreements mean the source_file
mapping is wrong, and they are counted and reported rather than passed off as clean data.

A pocket whose aggregate docking_results.json is missing (docking crashed on a bad molecule)
falls back to the per-ligand *_results.json written before the crash.

Usage:
    python scripts/consolidate_docking.py \
        --base results/vina_best_head1 --out results/vina_fixed_best_gen/vina_fixed_best_vina_docked.pt
"""
import argparse
import json
import os
import sys
from glob import glob

import numpy as np
import torch
from rdkit import Chem, RDLogger

sys.path.append(os.path.abspath('./'))  # same as the other scripts/: run from the repo root
from utils import reconstruct, transforms

RDLogger.DisableLog('rdApp.*')


def _entries_for_pocket(dock_dir):
    agg = os.path.join(dock_dir, 'docking_results.json')
    if os.path.isfile(agg):
        try:
            with open(agg) as f:
                return json.load(f), 'full'
        except Exception:
            pass
    per = sorted(glob(os.path.join(dock_dir, 'ligand_*_results.json')))
    out = []
    for pf in per:
        try:
            with open(pf) as f:
                out.append(json.load(f))
        except Exception:
            continue
    return out, ('partial' if out else 'empty')


def load_canonical_idx2full(sampling_dir, canonical):
    """pocket index -> ligand_filename, from targetdiff's canonical ordering (optional)."""
    path = os.path.join(sampling_dir, canonical)
    if not os.path.isfile(path):
        return {}
    try:
        obj = torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location='cpu')
    idx2full = {}
    for i, elem in enumerate(obj):
        for mol in (elem if isinstance(elem, list) else [elem]):
            lf = mol.get('ligand_filename') if isinstance(mol, dict) else None
            if lf:
                idx2full[i] = lf
                break
    return idx2full


class _RawCache:
    """Loads <base>/id{i}/<source_file> once each; several ligands can share a source file."""

    def __init__(self, base):
        self.base = base
        self._cache = {}

    def get(self, pocket_idx, source_file):
        key = (pocket_idx, source_file)
        if key not in self._cache:
            path = os.path.join(self.base, f'id{pocket_idx}', source_file)
            self._cache[key] = torch.load(path, map_location='cpu') if os.path.isfile(path) else None
        return self._cache[key]


def rebuild_mol(raw, atom_enc_mode='add_aromatic'):
    """raw sampler dict {'pos','v'} -> (mol, smiles, pos, v); (None, None, pos, v) if it fails.

    atom_affinity is filled with ones -- the same default dock_generated_ligands.py falls back to
    when the sampler stored no per-atom affinity (these runs store only a scalar `exp_on`), so the
    '_affinity_weight' atom property matches what the docking run itself saw.
    """
    if raw is None or 'pos' not in raw or 'v' not in raw:
        return None, None, None, None
    # float64 to match what targetdiff/kgdiff store. The sampler writes float32, which is also
    # exactly what OpenBabel's SetVector rejects (see utils/reconstruct.make_obmol).
    pos, v = np.asarray(raw['pos'], dtype=np.float64), np.asarray(raw['v'])
    try:
        z = transforms.get_atomic_number_from_index(v, mode=atom_enc_mode)
        aromatic = transforms.is_aromatic_from_index(v, mode=atom_enc_mode)
        mol = reconstruct.reconstruct_from_generated(pos, z, aromatic, np.ones_like(v))
        if mol is None or mol.GetNumAtoms() == 0:
            return None, None, pos, v
        return mol, Chem.MolToSmiles(mol), pos, v
    except Exception:
        return None, None, pos, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True, help='results/<model> with id{N}/docking_results/')
    ap.add_argument('--out', required=True, help='output .pt path')
    ap.add_argument('--n_pockets', type=int, default=100)
    ap.add_argument('--sampling_dir', default='./results/sampling_results')
    ap.add_argument('--canonical', default='targetdiff_vina_docked.pt')
    ap.add_argument('--atom_enc_mode', default='add_aromatic')
    ap.add_argument('--no_mol', action='store_true',
                    help='skip molecule reconstruction (scores only -- the old behaviour)')
    args = ap.parse_args()

    idx2full = load_canonical_idx2full(args.sampling_dir, args.canonical)
    raws = _RawCache(args.base)

    grouped = []
    n_full = n_partial = n_empty = n_mols = 0
    n_ok = n_fail = n_no_source = n_mismatch = 0

    for i in range(args.n_pockets):
        dock_dir = os.path.join(args.base, f'id{i}', 'docking_results')
        pocket = []
        if os.path.isdir(dock_dir):
            entries, kind = _entries_for_pocket(dock_dir)
            n_full += kind == 'full'
            n_partial += kind == 'partial'
            n_empty += kind == 'empty'
            for e in entries:
                rec = {
                    'vina': e.get('vina_results') or {},
                    'chem_results': e.get('chem_results') or {},
                    'ligand_filename': idx2full.get(i),
                }
                if not args.no_mol:
                    src = e.get('source_file')
                    mol = smiles = pos = v = None
                    if src:
                        mol, smiles, pos, v = rebuild_mol(raws.get(i, src), args.atom_enc_mode)
                    else:
                        n_no_source += 1
                    if mol is None:
                        n_fail += 1
                    else:
                        n_ok += 1
                        ref = e.get('smiles')
                        if ref and ref != smiles:
                            n_mismatch += 1
                    rec.update({'mol': mol, 'smiles': smiles, 'pred_pos': pos, 'pred_v': v})
                pocket.append(rec)
            n_mols += len(pocket)
        else:
            n_empty += 1
        grouped.append(pocket)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(grouped, args.out)

    print(f'{args.base}: pockets full={n_full} partial={n_partial} empty={n_empty} | '
          f'{n_mols} mols -> {args.out}')
    if not args.no_mol:
        print(f'  reconstructed mols : {n_ok}/{n_mols}   failed: {n_fail}')
        if n_no_source:
            print(f'  [WARN] {n_no_source} entries carry no "source_file" -> no molecule rebuilt')
        if n_mismatch:
            print(f'  [WARN] {n_mismatch} molecules disagree with the SMILES the docking run recorded '
                  f'-- the source_file mapping is WRONG for those; do not trust them')
        elif n_ok:
            print(f'  SMILES cross-check : all {n_ok} agree with the docking run')


if __name__ == '__main__':
    main()
