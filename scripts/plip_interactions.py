"""PLIP protein-ligand interaction profiling of GENERATED poses, one JSONL per pocket.

Why PLIP and not the Vina score: `vdw_loss_mode: vina` is TRAINED to minimise Vina's own
hbonding/hydrophobic terms, so scoring those molecules with Vina (or with the repo's
`utils/vina_rules.py` ramps) is circular. PLIP is an independent instrument -- it types atoms
chemically and applies real geometric criteria including ANGLES (Vina has no angular term at
all), so agreement between the two is evidence rather than tautology.

Poses: the GENERATED pose only. `m['mol']`'s conformer was verified to equal `m['pred_pos']`
exactly for every model; the re-docked poses live separately in `m['vina'][mode][0]['pose']`
as PDBQT text (3-7 A RMSD away, and with Vina-added hydrogens so the atom count differs).
Measuring interactions on a Vina-redocked pose would put Vina back in the loop, so those are
never touched.

Two receptor-file defects in `data/test_set` silently zero out PLIP if not handled; both were
found by smoke-testing NATIVE crystal ligands, which must never score zero:

  1. CHAIN IDs LIVE IN THE segID COLUMN.  Many CrossDocked receptors leave column 22 (chainID)
     degenerate -- e.g. GLMU_STRPN_2_459_0 is chain 'A' for all 10326 atoms -- while the real
     chain is in columns 73-76 (segID). OpenBabel keys residues on (chainID, resnum), so every
     physical chain collapses into one logical chain: in that file 3442 of 3442 atom keys
     collide, and in RG1_RAUSE_1_513_0 7544 of 7544 do. The merged "residues" then span several
     chains and their centroids land in the middle of the complex, up to 47 A from the ligand,
     so PLIP's binding-site selection (residue centroid within config.BS_DIST) finds NOTHING and
     reports 0 interactions with no error. `remap_chains()` rebuilds column 22 from the distinct
     (chainID, segID) pairs. Measured on the 100 native ligands: 11 pockets scored (0,0), 10 are
     fixed by the remap with BS_DIST left at its 7.5 A default, and the 5 control pockets are
     bit-identical before and after. (The 11th, CHIA_SERMA_19_563_0, is a true zero: its
     "ligand" is 1,4-dioxane -- 6 heavy atoms, no donors, and no hydrophobic carbons because
     every carbon neighbours an oxygen.)
     Raising BS_DIST instead is a band-aid: it recovers only 4 of the 11 and would need ~48 A --
     i.e. the whole protein as binding site -- to cover the worst case.

  2. RECEPTORS CONTAIN IONS.  The files are `protein or ion and not water` selections, so PLIP
     detects ZN/MG/etc. as additional ligands. `interaction_sets` must be keyed on our own
     'LIG:Z:1', never `list(...)[0]`, or the ion's (empty) interaction set gets read instead.

Usage:
    python scripts/plip_interactions.py --pt results/vina_fixed_best_gen/vina_fixed_best_vina_docked.pt \
        --tag vina_fixed_best --pockets 0-9 --out eval_plip
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

# chain letters for the remap; 'Z' is reserved for the ligand we append
_CHAIN_POOL = [c for c in string.ascii_uppercase + string.ascii_lowercase + string.digits if c != 'Z']


# --------------------------------------------------------------------------- inputs
def _torch_load(path):
    """Load a .pt holding RDKit Mol objects under either torch generation.

    These files were written in the kgdiff env (torch 1.11), whose `torch.load` has no
    `weights_only` kwarg. torch >= 2.6 defaults it to True and then refuses to unpickle
    `rdkit.Chem.rdchem.Mol`, so anything importing this module from a newer env dies on load.
    """
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:                                        # torch < 1.13: no such kwarg
        return torch.load(path, map_location='cpu')


def load_grouped(path):
    """Normalise the .pt layouts to list[pocket] of list[mol_dict].

    Same three layouts `scripts/eval_export_sdf.py:42` handles: grouped, flat, dict-wrapped.
    NATIVE (crossdocked_test_vina_docked.pt) is flat with one entry per pocket, so it becomes
    100 pockets of 1 molecule -- exactly what we want for the reference row.

    Flat layouts are placed at their CANONICAL index, never renumbered by order of appearance:
    index i must be the CrossDock data_id because the p<NNN>_m<MMMM> titles this script emits are
    joined against eval_out/<model>/ (docked_names.txt, posecheck/, manifest -> receptor). MolCRAFT's
    flat .pt appears in a different order than the canonical set (98/100 positions differ), so
    appearance order would label every molecule with the wrong pocket -- the same defect
    eval_export_sdf.py:68 fixes (it shifted 77 PIDiff pockets onto the wrong receptor).
    Alignment keys on the FULL ligand_filename, with a pocket-directory fallback only when that
    directory maps to exactly one index -- seven CrossDock sites appear twice under one directory
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
        # no canonical file -> appearance order (unchanged legacy behaviour)
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


def canonical_ligand_files(path='results/sampling_results/targetdiff_vina_docked.pt'):
    """Canonical ligand_filename per CrossDock data_id, from the canonical grouped set.

    Same source and semantics as eval_export_sdf.py:128 -- the two must agree or the SDF exports
    and the PLIP jsonl would disagree on what pocket i is.
    """
    if not os.path.isfile(path):
        return []
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
    """Per-sample layout: <root>/id<N>/result_*.pt holding raw `pos` / `v`, no RDKit mol.

    This is how the guidance-OFF runs are stored -- their consolidated `*_vina_docked.pt` carries
    only chem_results/ligand_filename/vina, with the geometry left behind in the per-sample files.
    The molecule is rebuilt exactly as results/analysis_scripts/analyse_noguide.py:126-130 does,
    so the two analyses stay comparable.

    The pocket index comes from the id<N> directory name; `ligand_filename` is recovered from the
    consolidated .pt, which shares that indexing.
    """
    # the repo root must be importable for utils.* -- the script may be run from anywhere
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
    """'<POCKET>/<pdb>_<ch>_rec_..._lig_....sdf' -> (pocket, receptor.pdb)  [eval_export_sdf.py:32]"""
    pocket = os.path.dirname(ligand_filename)
    stem = os.path.basename(ligand_filename).split('_rec')[0] + '_rec.pdb'
    return pocket, os.path.join(test_set_root, pocket, stem)


# --------------------------------------------------------------------------- receptor
def remap_chains(lines):
    """Rewrite column 22 (chainID) from the distinct (chainID, segID) pairs.

    See the module docstring, defect 1. Returns (lines, n_chains). A file that already has a
    single (chainID, segID) pair is returned untouched.
    """
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


# --------------------------------------------------------------------------- PLIP
def plip_profile(pdb_path):
    """Run PLIP on the complex and return the interaction profile of OUR ligand (LIG:Z:1).

    Keyed explicitly on 'LIG' -- the receptors carry ions that PLIP also reports as ligands
    (module docstring, defect 2).
    """
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

    # `pa` = the PROTEIN atom's original PDB index. Added so the interaction fingerprint can be
    # built at ATOM granularity, not just residue: DiffInt (JCIM 2025, 65, 71-82) defines
    # "reconstruction of hydrogen bonds" as reproducing the reference bond with the CORRECT
    # PROTEIN ATOM, whereas the residue-level bit collapses three contacts to one residue into a
    # single bit and is therefore more lenient. Both are emitted; nothing existing is removed.
    # For an H-bond the protein side is the donor when protisdon, else the acceptor.
    hb = [{'res': res(h), 'restype': h.restype, 'resnr': int(h.resnr), 'chain': h.reschain,
           'd_ad': round(float(h.distance_ad), 3), 'd_ah': round(float(h.distance_ah), 3),
           'angle': round(float(h.angle), 2), 'protisdon': bool(h.protisdon),
           'pa': int(h.d_orig_idx if h.protisdon else h.a_orig_idx)}
          for h in list(s.hbonds_pdon) + list(s.hbonds_ldon)]
    hy = [{'res': res(x), 'restype': x.restype, 'resnr': int(x.resnr), 'chain': x.reschain,
           'd': round(float(x.distance), 3), 'pa': int(x.bsatom_orig_idx)}
          for x in s.hydrophobic_contacts]
    # Atom-pair EDGE expansion for the group/ring-level interactions. PLIP reports a salt bridge per
    # charge-center pair and a pi-stack per ring pair (that is n_saltbridge / n_pistack). NCIDiff
    # (results/papers/NCIdiff.pdf) counts these on its bipartite protein-ligand graph as "a set
    # of edges between every possible pair from a ligand motif's atoms to a protein motif's atoms"
    # -- the CARTESIAN product |ligand-motif atoms| x |protein-motif atoms|. Verified to reproduce
    # NCIDiff's Reference exactly for pi-pi (9.32 == 9.32) and closely for salt bridge (4.26 vs
    # 4.81). Emitted alongside the object counts so external PLIP papers can be matched; nothing
    # existing changes.
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
    """Confounder controls: hydrophobic contact counts scale with size and greasiness, so a
    model can inflate them simply by making bigger, fattier molecules. These let the aggregate
    normalise per heavy atom and build a size-matched subset."""
    z = [a.GetAtomicNum() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    d = {'heavy': len(z),
         'pct_no': round(float(np.mean(np.isin(z, [7, 8]))), 4) if z else 0.0,
         'logp': np.nan, 'rings': -1, 'rotb': -1}
    # Crippen/ring descriptors need a sanitized molecule; chemically invalid generated molecules
    # (and the sanitize=False RemoveHs path) can fail here. heavy/pct_no are pure element counts
    # and always survive, so the size normalisation the aggregate depends on is never lost.
    try:
        d['logp'] = round(float(Crippen.MolLogP(mol)), 3)
        d['rings'] = int(rdmd.CalcNumRings(mol))
        d['rotb'] = int(rdmd.CalcNumRotatableBonds(mol))
    except Exception:                                          # noqa: BLE001
        pass
    return d


# --------------------------------------------------------------------------- worker
def _one(mol, prot_lines, q):
    """Child-process body: one molecule -> profile dict (or an error string)."""
    try:
        with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as tf:
            tmp = tf.name
        write_complex(prot_lines, mol, tmp)
        prof = plip_profile(tmp)
        os.unlink(tmp)
        q.put(('ok', prof))
    except Exception as e:                                    # noqa: BLE001 - report, never crash the pocket
        q.put(('err', f'{type(e).__name__}: {e}'))


def analyse_molecule(mol, prot_lines, timeout):
    """Run one molecule in a child process so a hang can actually be killed.

    PoseBusters taught this: a single pathological molecule stalls inside C++ where a Python
    signal cannot reach, and it took out whole pockets (see scripts/eval_pb_parallel.sh).
    OpenBabel is the same kind of risk, so the timeout has to be a process kill.
    """
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
                    help='per-sample dir (<root>/id<N>/result_*.pt) -- the guidance-OFF layout')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--out', default='eval_plip')
    ap.add_argument('--test_set', default='data/test_set')
    ap.add_argument('--pockets', default='', help='e.g. 0-9 or 3,7,11 (default: all)')
    ap.add_argument('--limit', type=int, default=0, help='max molecules per pocket (0 = all)')
    ap.add_argument('--timeout', type=float, default=120.0, help='seconds per molecule')
    args = ap.parse_args()

    if not args.pt and not args.samples:
        ap.error('--pt 또는 --samples 중 하나는 필요합니다')
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
                n_null += 1                                   # vina_fixed_best has 142 of these
                continue
            # Generated molecules can be chemically invalid -- vina_fixed_best contains
            # hypervalent nitrogens (neutral 4-valent N) that make RemoveHs raise
            # AtomValenceException. Anything RDKit throws here must cost us that one molecule,
            # never the pocket: an unguarded raise previously killed 2 whole array tasks
            # (12 pockets, ~1200 molecules) over 2 bad molecules.
            try:
                if any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()):   # 8 entries carry explicit H
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
