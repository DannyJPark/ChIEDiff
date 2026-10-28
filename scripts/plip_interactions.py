"""PLIP protein-ligand interaction profiling of generated poses, one JSONL per pocket.

PLIP stands in for the Vina score because the energy-guided model is trained to minimise Vina's
own hbonding and hydrophobic terms. PLIP types atoms chemically and applies geometric criteria
that include angles, which Vina has no term for, so agreement between the two is evidence rather
than tautology.

Only the generated pose is profiled. The conformer in m['mol'] equals m['pred_pos'] exactly.
Re-docked poses sit separately in m['vina'][mode][0]['pose'] as PDBQT text, 3 to 7 A RMSD away
and with Vina-added hydrogens, and are never touched, because measuring interactions on them
would put Vina back in the loop.

Two receptor-file defects in the CrossDocked test set silently zero out PLIP if they are not
handled. Both show up when crystal ligands are profiled as a smoke test, since those must never
score zero.

  1. Chain ids live in the segID column, 73 to 76, while column 22 (chainID) is often
     degenerate. GLMU_STRPN_2_459_0 is chain 'A' for all 10326 atoms. OpenBabel keys residues
     on (chainID, resnum), so every physical chain collapses into one and all 3442 atom keys in
     that file collide, as do all 7544 in RG1_RAUSE_1_513_0. The merged residues then have
     centroids in the middle of the complex, up to 47 A from the ligand, so PLIP's binding-site
     selection, which takes residues whose centroid falls within config.BS_DIST, finds nothing
     and reports 0 interactions with no error. remap_chains() rebuilds column 22 from the
     distinct (chainID, segID) pairs. Over the 100 crystal ligands 11 pockets scored (0,0), the
     remap fixes 10 of them with BS_DIST left at its 7.5 A default, and 5 control pockets are
     bit-identical before and after. The 11th, CHIA_SERMA_19_563_0, is a true zero, its ligand
     being 1,4-dioxane with 6 heavy atoms, no donors and no hydrophobic carbons. Raising BS_DIST
     instead recovers only 4 of the 11 and would need about 48 A, the whole protein, to cover
     the worst case.

  2. The receptor files are protein-or-ion selections, so PLIP reports ZN, MG and the like as
     additional ligands. interaction_sets must be keyed on the ligand this script appends,
     LIG:Z:1, and never on the first key, or an ion's empty interaction set is read instead.
"""
import argparse
import json
import multiprocessing as mp
import os
import string
import sys
import tempfile

import glob

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, rdMolDescriptors as rdmd

RDLogger.DisableLog('rdApp.*')

# chain letters for the remap. 'Z' is reserved for the ligand this script appends.
_CHAIN_POOL = [c for c in string.ascii_uppercase + string.ascii_lowercase + string.digits if c != 'Z']


def _torch_load(path):
    """Load a .pt holding RDKit Mol objects under either torch generation. torch 2.6 and later
    default weights_only to True and then refuse to unpickle rdkit.Chem.rdchem.Mol, while torch
    1.11 has no such kwarg, so the kwarg has to be tried first."""
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:                                        # torch < 1.13: no such kwarg
        return torch.load(path, map_location='cpu')


def load_grouped(path):
    """Normalise a grouped, flat or dict-wrapped .pt to list[pocket] of list[mol_dict]. A
    crystal-ligand reference file is flat with one entry per pocket and becomes 100 pockets of
    one molecule, which is what the reference row needs. Flat layouts go to their canonical index and are never renumbered by order of appearance.
    Index i must be the CrossDocked data_id, because the p<NNN>_m<MMMM> titles this script emits
    are joined against the posecheck results and the pocket-to-receptor manifest. Some released
    sample files are in a different order than the canonical set, in one case with 98 of 100
    positions differing.

    Alignment keys on the full ligand_filename, with a pocket-directory fallback only when that
    directory maps to exactly one index. Seven CrossDocked sites appear twice under one directory
    with different reference ligands, so a directory-keyed map would duplicate molecules.
    """
    obj = _torch_load(path)
    if isinstance(obj, dict):
        obj = obj.get('all_results', obj.get('results', []))
    if not obj:
        return []
    if isinstance(obj[0], list):
        return obj
    if isinstance(obj[0], dict) and 'ligand_filename' in obj[0]:
        canon = canonical_ligand_files()
        if canon:
            full2idx = {lf: i for i, lf in enumerate(canon) if lf}
            dir_counts = {}
            for lf in canon:
                if lf:
                    d = os.path.dirname(lf)
                    dir_counts[d] = dir_counts.get(d, 0) + 1
            dir2idx = {os.path.dirname(lf): i for i, lf in enumerate(canon)
                       if lf and dir_counts[os.path.dirname(lf)] == 1}
            out, n_unmatched = [[] for _ in canon], 0
            for e in obj:
                lf = e.get('ligand_filename')
                if lf is None:
                    continue
                i = full2idx.get(lf)
                if i is None:
                    i = dir2idx.get(os.path.dirname(lf))   # unique-dir fallback only
                if i is None:
                    n_unmatched += 1
                    continue
                out[i].append(e)
            if n_unmatched:
                print(f'[warn] {n_unmatched} molecule(s) matched no canonical pocket, dropped')
            return out
        # no canonical file -> fall back to appearance order
        groups, order = {}, []
        for e in obj:
            k = e.get('ligand_filename')
            if k is None:
                continue
            if k not in groups:
                groups[k] = []
                order.append(k)
            groups[k].append(e)
        return [groups[k] for k in order]
    return [[e] for e in obj]


def canonical_ligand_files(path='data/test_pocket_order.json'):
    """The canonical ligand_filename of each test pocket, in pocket order. This has to agree with
    eval_export_sdf.canonical_ligand_files, or the SDF exports and this script's jsonl disagree
    on which pocket is pocket i."""
    if not os.path.isfile(path):
        return []
    if path.endswith('.json'):
        with open(path) as fh:
            return json.load(fh)
    out = []
    for elem in _torch_load(path):
        mols = elem if isinstance(elem, list) else [elem]
        lf = None
        for m in mols:
            if isinstance(m, dict) and m.get('ligand_filename'):
                lf = m['ligand_filename']
                break
        out.append(lf)
    return out


def load_samples(root, test_set_root):
    """Read the per-sample layout <root>/id<N>/result_*.pt, which holds raw pos and v and no
    RDKit mol, the consolidated .pt for that run carrying everything but the geometry. The
    molecule is rebuilt through the same reconstruction path the sampler's own export uses, so a
    per-sample run stays comparable with a grouped one. The pocket index comes from the id<N>
    directory name and ligand_filename from the consolidated .pt, which shares it.
    """
    # The repo root has to be importable, because the script may be run from anywhere.
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
    from utils import reconstruct
    from utils import transforms as _t

    lf = {}
    cons = glob.glob(os.path.join(root, '*_vina_docked.pt'))
    if cons:
        obj = _torch_load(cons[0])
        if isinstance(obj, dict):
            obj = obj.get('all_results', obj.get('results', []))
        for pi, grp in enumerate(obj):
            g = grp if isinstance(grp, list) else [grp]
            if g and g[0].get('ligand_filename'):
                lf[pi] = g[0]['ligand_filename']

    out = {}
    for d in sorted(glob.glob(os.path.join(root, 'id*'))):
        m = os.path.basename(d)[2:]
        if not m.isdigit():
            continue
        pi = int(m)
        if pi not in lf:
            continue
        mols = []
        for f in sorted(glob.glob(os.path.join(d, 'result_*.pt')),
                        key=lambda p: int(os.path.basename(p)[7:-3])):
            try:
                r = _torch_load(f)
                pos = np.asarray(r['pos'], dtype=np.float64)
                v = np.asarray(r['v'])
                if not np.all(np.isfinite(pos)):
                    continue
                at = _t.get_atomic_number_from_index(v, mode='add_aromatic')
                ar = _t.is_aromatic_from_index(v, mode='add_aromatic')
                mol = reconstruct.reconstruct_from_generated(pos, at, ar, np.ones_like(v))
                Chem.SanitizeMol(mol)
                if len(Chem.GetMolFrags(mol)) > 1:      # keep single-fragment molecules only
                    continue
            except Exception:                            # noqa: BLE001
                continue
            mols.append({'mol': mol, 'ligand_filename': lf[pi],
                         'smiles': Chem.MolToSmiles(mol)})
        out[pi] = mols
    return [out.get(i, []) for i in range(max(out) + 1)] if out else []


def paths_for(ligand_filename, test_set_root):
    """Map a ligand_filename to (pocket, receptor pdb), as eval_export_sdf.paths_for does."""
    pocket = os.path.dirname(ligand_filename)
    stem = os.path.basename(ligand_filename).split('_rec')[0] + '_rec.pdb'
    return pocket, os.path.join(test_set_root, pocket, stem)


def remap_chains(lines):
    """Rewrite column 22 (chainID) from the distinct (chainID, segID) pairs and return
    (lines, n_chains). A file with a single pair is returned untouched. See the first
    receptor-file defect in the module docstring."""
    pairs = []
    for l in lines:
        k = (l[21], l[72:76].strip())
        if k not in pairs:
            pairs.append(k)
    if len(pairs) <= 1:
        return lines, len(pairs)
    m = {k: _CHAIN_POOL[i % len(_CHAIN_POOL)] for i, k in enumerate(pairs)}
    return [l[:21] + m[(l[21], l[72:76].strip())] + l[22:] for l in lines], len(pairs)


def load_receptor(rec_path):
    """Read ATOM/HETATM records, pad to 80 cols, and remap chains. Cached per pocket."""
    with open(rec_path) as f:
        lines = [l.rstrip('\n').ljust(80) for l in f if l.startswith(('ATOM', 'HETATM'))]
    return remap_chains(lines)


def write_complex(prot_lines, mol, out_path):
    """protein + ligand (HETATM, resname LIG, chain Z, resnum 1) -> one PDB PLIP can read.

    Ligand atom serials continue from the protein's maximum so they cannot collide.
    """
    mx = max(int(l[6:11]) for l in prot_lines)
    lig = []
    for i, l in enumerate(Chem.MolToPDBBlock(mol, flavor=4).splitlines()):
        if not l.startswith(('ATOM', 'HETATM')):
            continue
        l = l.ljust(80)
        lig.append('HETATM' + str(mx + 1 + i).rjust(5) + l[11:17] + 'LIG' + ' Z   1' + l[26:])
    with open(out_path, 'w') as f:
        f.write('\n'.join(prot_lines + lig) + '\nEND\n')


def plip_profile(pdb_path):
    """Interaction profile of the appended ligand, keyed explicitly on LIG because the receptors
    carry ions that PLIP also reports as ligands."""
    from plip.structure.preparation import PDBComplex
    c = PDBComplex()
    c.load_pdb(pdb_path)
    c.analyze()
    keys = [k for k in c.interaction_sets if k.startswith('LIG')]
    if not keys:
        return None
    s = c.interaction_sets[keys[0]]

    def res(x):
        return f'{x.restype}{x.resnr}{x.reschain}'

    # pa is the protein atom's original PDB index, which lets the fingerprint be built at atom
    # granularity as well as residue granularity. DiffInt (JCIM 2025, 65, 71-82) requires a
    # reconstructed hydrogen bond to name the correct protein atom, whereas a residue-level bit
    # collapses three contacts to one residue into a single bit and is more lenient. Both
    # granularities are emitted. For an H-bond the protein side is the donor when protisdon.
    hb = [{'res': res(h), 'restype': h.restype, 'resnr': int(h.resnr), 'chain': h.reschain,
           'd_ad': round(float(h.distance_ad), 3), 'd_ah': round(float(h.distance_ah), 3),
           'angle': round(float(h.angle), 2), 'protisdon': bool(h.protisdon),
           'pa': int(h.d_orig_idx if h.protisdon else h.a_orig_idx)}
          for h in list(s.hbonds_pdon) + list(s.hbonds_ldon)]
    hy = [{'res': res(x), 'restype': x.restype, 'resnr': int(x.resnr), 'chain': x.reschain,
           'd': round(float(x.distance), 3), 'pa': int(x.bsatom_orig_idx)}
          for x in s.hydrophobic_contacts]
    # PLIP reports one salt bridge per charge-centre pair and one pi-stack per ring pair, which
    # is what n_saltbridge and n_pistack hold. NCIDiff instead counts the cartesian product of
    # ligand-motif and protein-motif atoms, so that expansion is emitted alongside the object
    # counts to make PLIP numbers from other papers comparable. It reproduces NCIDiff's reference
    # value exactly for pi-pi, 9.32 against 9.32, and closely for salt bridges, 4.26 against 4.81.
    sb_edges = sum(len(sb.positive.atoms) * len(sb.negative.atoms)
                   for sb in list(s.saltbridge_lneg) + list(s.saltbridge_pneg))
    pp_edges = sum(len(pp.ligandring.atoms) * len(pp.proteinring.atoms) for pp in s.pistacking)
    return {
        'hbond': hb,
        'hydrophobic': hy,
        'n_hbond': len(hb),
        'n_hydrophobic': len(hy),
        'n_pistack': len(s.pistacking),
        'n_saltbridge': len(s.saltbridge_lneg) + len(s.saltbridge_pneg),
        'n_pistack_edges': int(pp_edges),
        'n_saltbridge_edges': int(sb_edges),
        'n_halogen': len(s.halogen_bonds),
        'n_waterbridge': len(s.water_bridges),
        'n_bs_atoms': len(s.bindingsite.all_atoms),
    }


def descriptors(mol):
    """Confounder controls. Hydrophobic contact counts scale with size and greasiness, so these
    columns let the aggregate normalise per heavy atom and build a size-matched subset."""
    z = [a.GetAtomicNum() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    d = {'heavy': len(z),
         'pct_no': round(float(np.mean(np.isin(z, [7, 8]))), 4) if z else 0.0,
         'logp': np.nan, 'rings': -1, 'rotb': -1}
    # The Crippen and ring descriptors need a sanitized molecule and can fail on a chemically
    # invalid one. heavy and pct_no are pure element counts and always survive, so the size
    # normalisation the aggregate depends on is never lost.
    try:
        d['logp'] = round(float(Crippen.MolLogP(mol)), 3)
        d['rings'] = int(rdmd.CalcNumRings(mol))
        d['rotb'] = int(rdmd.CalcNumRotatableBonds(mol))
    except Exception:                                          # noqa: BLE001
        pass
    return d


def _one(mol, prot_lines, q):
    """Child-process body returning one profile dict or an error string."""
    try:
        with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as tf:
            tmp = tf.name
        write_complex(prot_lines, mol, tmp)
        prof = plip_profile(tmp)
        os.unlink(tmp)
        q.put(('ok', prof))
    except Exception as e:                                    # noqa: BLE001 report, never crash the pocket
        q.put(('err', f'{type(e).__name__}: {e}'))


def analyse_molecule(mol, prot_lines, timeout):
    """Run one molecule in a child process so a hang can be killed. A single pathological
    molecule can stall inside OpenBabel, which PLIP calls, where a Python signal never gets a
    chance to run, so the timeout has to be a process kill."""
    q = mp.Queue()
    p = mp.Process(target=_one, args=(mol, prot_lines, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return 'timeout', None
    if q.empty():
        return 'died', None
    return q.get()


def parse_pockets(spec, n):
    if not spec:
        return list(range(n))
    out = []
    for part in spec.split(','):
        if '-' in part:
            a, b = part.split('-')
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return [i for i in out if i < n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pt', default='', help='grouped/flat .pt with RDKit mols')
    ap.add_argument('--samples', default='',
                    help='per-sample dir (<root>/id<N>/result_*.pt), the raw sampler layout')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--out', default='eval_plip')
    ap.add_argument('--test_set', default='data/test_set')
    ap.add_argument('--pockets', default='', help='e.g. 0-9 or 3,7,11 (default: all)')
    ap.add_argument('--limit', type=int, default=0, help='max molecules per pocket (0 = all)')
    ap.add_argument('--timeout', type=float, default=120.0, help='seconds per molecule')
    args = ap.parse_args()

    if not args.pt and not args.samples:
        ap.error('one of --pt or --samples is required')
    data = load_samples(args.samples, args.test_set) if args.samples else load_grouped(args.pt)
    outdir = os.path.join(args.out, args.tag)
    os.makedirs(outdir, exist_ok=True)

    for pi in parse_pockets(args.pockets, len(data)):
        mols = data[pi]
        if not mols:
            continue
        pocket, rec = paths_for(mols[0]['ligand_filename'], args.test_set)
        if not os.path.exists(rec):
            print(f'[WARN] pocket {pi:03d} ({pocket}): receptor missing -> {rec}; skipping', flush=True)
            continue
        prot_lines, n_chains = load_receptor(rec)

        rows, n_null, n_to, n_err = [], 0, 0, 0
        todo = mols[:args.limit] if args.limit else mols
        for mi, e in enumerate(todo):
            mol = e.get('mol')
            if mol is None or mol.GetNumConformers() == 0:
                n_null += 1                                   # counted, never silently dropped
                continue
            # A generated molecule can be chemically invalid, a neutral four-valent nitrogen
            # making RemoveHs raise AtomValenceException. Anything RDKit throws here must cost
            # one molecule and never the rest of the pocket.
            try:
                if any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()):   # some entries carry explicit H
                    mol = Chem.RemoveHs(mol, sanitize=False)
                desc = descriptors(mol)
            except Exception:                                 # noqa: BLE001
                n_err += 1
                continue
            status, prof = analyse_molecule(mol, prot_lines, args.timeout)
            if status == 'timeout':
                n_to += 1
                continue
            if status != 'ok' or prof is None:
                n_err += 1
                continue
            rows.append({'pk': e['ligand_filename'], 'pocket': pocket, 'pocket_idx': pi, 'mi': mi,
                         'smiles': e.get('smiles'), **desc, **prof})

        with open(os.path.join(outdir, f'pocket{pi:03d}.jsonl'), 'w') as f:
            for r in rows:
                f.write(json.dumps(r) + '\n')
        # never let a partial pocket read as complete
        if n_to or n_err or n_null:
            with open(os.path.join(outdir, f'pocket{pi:03d}.skipped'), 'w') as f:
                f.write(f'null_mol={n_null}\ntimeout={n_to}\nerror={n_err}\ntotal_in={len(todo)}\n')
        hb = np.mean([r['n_hbond'] for r in rows]) if rows else 0
        hy = np.mean([r['n_hydrophobic'] for r in rows]) if rows else 0
        print(f'pocket {pi:03d} {pocket:34s} chains={n_chains} n={len(rows):4d} '
              f'hb/mol={hb:5.2f} hy/mol={hy:5.2f} '
              f'(null={n_null} timeout={n_to} err={n_err})', flush=True)


if __name__ == '__main__':
    mp.set_start_method('fork', force=True)
    sys.exit(main())
