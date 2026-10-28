#!/usr/bin/env python3
"""ProLIF interaction counts for one pocket, recording both counting conventions per type.

Why this is a separate script from eval_posecheck.py: that script sums `int(bool(cell))`, which
is one count per (residue, interaction-type) and the ProLIF 2.2 default of `count=False`. The
PoseCheck paper and FlexSBDD (arXiv, Appendix B.4 Table 7) instead report the number of
individual contacts, the raw sum under `count=True`, which runs about three times higher for
hydrophobic contacts. Measured on one test pocket the two conventions give 2.36 and 5.29.
Comparing against those papers needs the per-contact count, which eval_posecheck.py does not
store. This script recomputes the fingerprint with count=True and writes, per molecule and per
interaction type, both the per-contact count (`<type>_contact`) and the per-residue presence
(`<type>_res`, which equals the boolean column eval_posecheck.py writes and is kept for
validation).

It computes the interaction fingerprint only, with no clashes and no strain, so it is far faster
than a full PoseCheck run, where the strain relaxation dominates. The receptor is still
protonated with PoseCheck's own loader and the same two prolif patches eval_posecheck.py applies,
so H-bond and metal handling are identical.

Run with the ged-posecheck interpreter:
    "$CONDA_BASE/envs/ged-posecheck/bin/python" scripts/eval_prolif_counts.py \
        --dir eval_out/mymodel --pocket 0

Input  : <dir>/manifest.csv + <dir>/sdf/pocket<NNN>.sdf
Output : <dir>/prolif_counts/pocket<NNN>.csv   molecule,<type>_contact,<type>_res,...
"""
import argparse
import csv
import multiprocessing as mp
import os
import sys
import time

from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')

_ENV_BIN = os.path.dirname(os.path.abspath(sys.executable))
os.environ['PATH'] = _ENV_BIN + os.pathsep + os.environ.get('PATH', '')

# VdWContact radii for real metals, the same table eval_posecheck.py uses, so that a
# metalloprotein pocket does not abort at fingerprint time.
_VDW_RADII = {'Co': 2.00, 'Zn': 1.39, 'Fe': 1.80, 'Mn': 1.80, 'Ni': 1.63, 'Cu': 1.40,
              'Mg': 1.73, 'Ca': 2.31, 'Na': 2.27, 'K': 2.75, 'Cd': 1.58, 'Hg': 1.55}
_VDW_RADII.update({k.upper(): v for k, v in list(_VDW_RADII.items())})


def _patch_prolif_vdwradii():
    """Identical to eval_posecheck.py. Fixes MDAnalysis protein loading for atoms whose PDB name
    starts "CO" and for receptors numbering residues from a negative value."""
    import MDAnalysis as mda
    import prolif as plf
    from posecheck.utils import loading
    extra = {'CO': 1.70, 'CD': 1.70, 'NA': 2.27, 'FE': 1.80, 'ZN': 1.39,
             'MG': 1.73, 'MN': 1.80, 'CA': 2.31, 'CU': 1.40, 'NI': 1.63, 'K': 2.75}

    def load_protein_prolif(protein_path):
        u = mda.Universe(protein_path)
        u.atoms.guess_bonds(vdwradii=extra)
        try:
            resids = u.residues.resids
            if resids.min() < 1:
                u.residues.resids = resids - int(resids.min()) + 1
        except Exception:
            pass
        return plf.Molecule.from_mda(u, NoImplicit=False)

    loading.load_protein_prolif = load_protein_prolif


_patch_prolif_vdwradii()


def _fingerprint(count):
    import prolif as plf
    return plf.Fingerprint(count=count, parameters={'VdWContact': {'vdwradii': _VDW_RADII}})


_PC = None


def _eval_one(mol, q):
    """Child process: count=True fingerprint of one ligand -> {type: (contact_sum, residue_count)}."""
    try:
        _PC.load_ligands_from_mols([mol])
        fp = _fingerprint(count=True)
        fp.run_from_iterable(_PC.ligands, _PC.protein)
        df = fp.to_dataframe()
        contact, res = {}, {}
        for col in df.columns:
            itype = col[-1] if isinstance(col, tuple) else str(col)
            v = int(df.iloc[0][col])                 # count=True -> integer number of contacts
            contact[itype] = contact.get(itype, 0) + v
            res[itype] = res.get(itype, 0) + int(v > 0)
        q.put((contact, res))
    except Exception as e:                           # noqa: BLE001
        q.put(('ERROR', repr(e)[:200]))


def evaluate_pocket(pc, mols, names, timeout, workers):
    global _PC
    _PC = pc
    ctx = mp.get_context('fork')
    rows, timed_out, failed = [], [], []
    running = {}
    todo = list(zip(names, mols))
    while todo or running:
        while todo and len(running) < workers:
            name, mol = todo.pop(0)
            q = ctx.Queue()
            p = ctx.Process(target=_eval_one, args=(mol, q))
            p.start()
            running[p] = (name, q, time.time() + timeout)
        time.sleep(0.02)
        for p in list(running):
            name, q, deadline = running[p]
            if p.is_alive() and time.time() < deadline:
                continue
            if p.is_alive():
                p.terminate(); p.join(5)
                if p.is_alive():
                    p.kill(); p.join()
                timed_out.append(name)
            else:
                try:
                    res = q.get_nowait()
                except Exception:
                    res = None
                if res is None:
                    failed.append((name, 'no result'))
                elif res[0] == 'ERROR':
                    failed.append((name, res[1]))
                else:
                    contact, residue = res
                    row = {'molecule': name}
                    for k, v in contact.items():
                        row[f'{k}_contact'] = v
                    for k, v in residue.items():
                        row[f'{k}_res'] = v
                    rows.append(row)
            p.join()
            del running[p]
    return rows, timed_out, failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--pocket', type=int, required=True)
    ap.add_argument('--timeout', type=int, default=60)
    ap.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    args = ap.parse_args()

    tag = f'{args.pocket:03d}'
    sdf_path = os.path.join(args.dir, 'sdf', f'pocket{tag}.sdf')
    out_path = os.path.join(args.dir, 'prolif_counts', f'pocket{tag}.csv')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    receptor = None
    with open(os.path.join(args.dir, 'manifest.csv')) as f:
        for row in csv.DictReader(f):
            if int(row['pocket_idx']) == args.pocket:
                receptor = row['receptor']
                break
    if receptor is None or not os.path.isfile(receptor):
        print(f'[skip] pocket {args.pocket}: receptor not found ({receptor})'); return
    if not os.path.isfile(sdf_path):
        print(f'[skip] pocket {args.pocket}: missing {sdf_path}'); return

    mols, names = [], []
    for m in Chem.SDMolSupplier(sdf_path, sanitize=False, removeHs=False):
        if m is None:
            continue
        mols.append(m)
        names.append(m.GetProp('_Name') if m.HasProp('_Name') else f'p{tag}_m{len(names):04d}')
    if not mols:
        print(f'[skip] pocket {args.pocket}: no molecules'); return

    from posecheck import PoseCheck
    pc = PoseCheck()
    pc.load_protein_from_pdb(receptor)

    rows, timed_out, failed = evaluate_pocket(pc, mols, names, args.timeout, args.workers)

    cols = ['molecule'] + sorted({k for r in rows for k in r if k != 'molecule'})
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, 0) for c in cols})
    if timed_out:
        with open(out_path + '.timeouts', 'w') as f:
            f.write('\n'.join(timed_out) + '\n')
    print(f'pocket {args.pocket}: {len(rows)} molecules, {len(timed_out)} timeout, {len(failed)} failed')


if __name__ == '__main__':
    main()
