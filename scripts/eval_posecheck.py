#!/usr/bin/env python3
"""
PoseCheck evaluation of one pocket: clashes, strain energy, and protein-ligand interactions.

These are the three metrics the CBYG paper reports in its Table 5 (labelled "Posebusters" there, but
its own text and reference [24] identify them as PoseCheck). They are NOT what our existing
eval_out/*/posebusters/ CSVs measure: PoseBusters' `internal_steric_clash` is a boolean on
INTRA-ligand overlap and its `internal_energy` is a pass/fail, whereas PoseCheck counts
PROTEIN-ligand clashes and returns a strain energy in kcal/mol.

Run from the repo root with the `posecheck` env's interpreter -- its rdkit is far newer than the
project's and must not leak into `kgdiff`:

    /home/ktori1361/anaconda3/envs/posecheck/bin/python scripts/eval_posecheck.py \
        --dir eval_out/vina_fixed --pocket 0

Input  : <dir>/manifest.csv (pocket_idx,pocket,receptor,n_mols,ref_ligand) + <dir>/sdf/pocket<NNN>.sdf
Output : <dir>/posecheck/pocket<NNN>.csv          molecule,clashes,strain_energy,n_interactions,<per-type>
         <dir>/posecheck/pocket<NNN>.csv.timeouts one molecule name per line (only if any timed out)

Concurrency model
-----------------
Loading the receptor costs ~13 s (PoseCheck shells out to `reduce` to protonate it), while a
molecule costs ~6 s -- dominated by the strain-energy force-field relaxation. So the receptor is
loaded ONCE in the parent and molecules are evaluated in forked children, which inherit the loaded
PoseCheck object copy-on-write: no reload, and a molecule that hangs the FF relaxation only kills
its own child. That hazard is real -- the same class of relaxation already stalled whole pockets in
the PoseBusters runs -- so every child gets a hard timeout and timed-out molecules are recorded in a
sidecar file rather than silently dropped.
"""
import argparse
import csv
import multiprocessing as mp
import os
import sys
import time

from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')

# PoseCheck protonates the receptor by shelling out to `hydride` (it hardcodes the command and
# ignores its own reduce_path argument), so that binary must be on PATH. We are invoked by absolute
# interpreter path rather than `conda activate` -- deliberately, to keep this env's much newer rdkit
# out of the project env -- which leaves the env's bin/ off PATH. Put it back, for this process only.
_ENV_BIN = os.path.join(os.path.dirname(os.path.abspath(sys.executable)))
os.environ['PATH'] = _ENV_BIN + os.pathsep + os.environ.get('PATH', '')

# A handful of CrossDock receptors (e.g. 2v3r, 5b08, 4keu) carry atoms whose PDB name starts "CO",
# which MDAnalysis reads as the element cobalt and then aborts protein loading with
# "vdw radii for types: CO". PoseCheck's load_protein_prolif calls MDAnalysis with no vdwradii
# override, so patch it to supply carbon's radius for CO (these are carbons) plus a few other
# metals that show up in receptors, keeping the whole molecule instead of losing the pocket.
def _patch_prolif_vdwradii():
    import MDAnalysis as mda
    import prolif as plf
    from posecheck.utils import loading
    extra = {'CO': 1.70, 'CD': 1.70, 'NA': 2.27, 'FE': 1.80, 'ZN': 1.39,
             'MG': 1.73, 'MN': 1.80, 'CA': 2.31, 'CU': 1.40, 'NI': 1.63, 'K': 2.75}

    def load_protein_prolif(protein_path):
        u = mda.Universe(protein_path)
        # prolif triggers guess_bonds() internally with no vdwradii, which is what raises on CO;
        # precompute the bonds here with the override so the internal call finds them already set.
        u.atoms.guess_bonds(vdwradii=extra)
        # Some receptors number residues from a negative value (e.g. -2); prolif casts residue
        # numbers to uint32 and overflows. Shift resids so the minimum is 1 -- identity is unchanged,
        # only the integer label moves.
        try:
            resids = u.residues.resids
            if resids.min() < 1:
                u.residues.resids = resids - int(resids.min()) + 1
        except Exception:
            pass
        return plf.Molecule.from_mda(u, NoImplicit=False)

    loading.load_protein_prolif = load_protein_prolif


_patch_prolif_vdwradii()


# The patch above only fixes PROTEIN LOADING (MDAnalysis guess_bonds). prolif's VdWContact
# interaction keeps its OWN radius table, so a receptor with a real metal (4keu / pocket 027 has
# two cobalt ions in its binuclear centre) still aborts every molecule at calculate_interactions()
# with "van der Waals radius for atom 'Co' not found", leaving an empty pocket CSV. PoseCheck
# builds its fingerprint as a bare plf.Fingerprint(), so supply the metal radii via `parameters`.
# Purely additive: an atom whose radius was already known is unaffected, so molecules that used to
# succeed stay bit-identical -- only ones that used to RAISE now produce a result.
def _patch_prolif_vdwcontact():
    import prolif as plf
    import posecheck.posecheck as _pcmod
    from posecheck.utils import interactions as _inter
    # element-symbol keys (prolif reports 'Co'); upper-case duplicates for safety
    radii = {'Co': 2.00, 'Zn': 1.39, 'Fe': 1.80, 'Mn': 1.80, 'Ni': 1.63, 'Cu': 1.40,
             'Mg': 1.73, 'Ca': 2.31, 'Na': 2.27, 'K': 2.75, 'Cd': 1.58, 'Hg': 1.55}
    radii.update({k.upper(): v for k, v in list(radii.items())})

    def generate_interaction_df(prot, lig):
        fp = plf.Fingerprint(parameters={'VdWContact': {'vdwradii': radii}})
        fp.run_from_iterable(lig, prot)
        return fp.to_dataframe()

    _inter.generate_interaction_df = generate_interaction_df
    _pcmod.generate_interaction_df = generate_interaction_df   # imported by name at posecheck.py:10


_patch_prolif_vdwcontact()

# Loaded once in the parent; forked children inherit it without re-running `reduce`.
_PC = None


def _eval_one(mol, q):
    """Child process: evaluate a single molecule against the inherited receptor."""
    try:
        _PC.load_ligands_from_mols([mol])
        clashes = _PC.calculate_clashes()[0]
        strain = _PC.calculate_strain_energy()[0]
        inter = _PC.calculate_interactions()
        # prolif returns a one-row frame whose columns are (ligand, residue, interaction-type);
        # collapse to a total plus a per-type breakdown so we can see WHICH interactions change.
        per_type = {}
        for col in inter.columns:
            itype = col[-1] if isinstance(col, tuple) else str(col)
            per_type[itype] = per_type.get(itype, 0) + int(bool(inter.iloc[0][col]))
        q.put((clashes, strain, int(sum(per_type.values())), per_type))
    except Exception as e:                      # a molecule PoseCheck cannot handle
        q.put(('ERROR', repr(e)[:200], None, None))


def evaluate_pocket(pc, mols, names, timeout, workers):
    """Evaluate every molecule with bounded concurrency and a per-molecule timeout.

    Returns (rows, timed_out_names, failed). `rows` are dicts; molecules that time out or raise are
    excluded from `rows` and reported separately so they are never silently averaged away.
    """
    global _PC
    _PC = pc
    ctx = mp.get_context('fork')

    rows, timed_out, failed = [], [], []
    running = {}          # proc -> (name, queue, deadline)
    todo = list(zip(names, mols))

    while todo or running:
        while todo and len(running) < workers:
            name, mol = todo.pop(0)
            q = ctx.Queue()
            p = ctx.Process(target=_eval_one, args=(mol, q))
            p.start()
            running[p] = (name, q, time.time() + timeout)

        time.sleep(0.05)
        for p in list(running):
            name, q, deadline = running[p]
            if p.is_alive() and time.time() < deadline:
                continue
            if p.is_alive():                      # over the deadline -> kill it
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
                    clashes, strain, n_inter, per_type = res
                    rows.append({'molecule': name, 'clashes': clashes,
                                 'strain_energy': strain, 'n_interactions': n_inter,
                                 **{f'int_{k}': v for k, v in per_type.items()}})
            p.join()
            del running[p]

    return rows, timed_out, failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True, help='eval_out/<model> (holds manifest.csv and sdf/)')
    ap.add_argument('--pocket', type=int, required=True, help='pocket_idx from manifest.csv')
    ap.add_argument('--timeout', type=int, default=300, help='per-molecule timeout (s)')
    ap.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    args = ap.parse_args()

    tag = f'{args.pocket:03d}'
    sdf_path = os.path.join(args.dir, 'sdf', f'pocket{tag}.sdf')
    out_path = os.path.join(args.dir, 'posecheck', f'pocket{tag}.csv')
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

    # sanitize=False so a molecule rdkit dislikes still reaches PoseCheck, as in the other workers
    mols, names = [], []
    for m in Chem.SDMolSupplier(sdf_path, sanitize=False, removeHs=False):
        if m is None:
            continue
        mols.append(m)
        names.append(m.GetProp('_Name') if m.HasProp('_Name') else f'p{tag}_m{len(names):04d}')
    if not mols:
        print(f'[skip] pocket {args.pocket}: no molecules in {sdf_path}'); return

    from posecheck import PoseCheck
    t0 = time.time()
    pc = PoseCheck()
    pc.load_protein_from_pdb(receptor)
    t_prot = time.time() - t0

    rows, timed_out, failed = evaluate_pocket(pc, mols, names, args.timeout, args.workers)

    # union of per-type interaction columns across molecules (a molecule may show none of a type)
    int_cols = sorted({k for r in rows for k in r if k.startswith('int_')})
    cols = ['molecule', 'clashes', 'strain_energy', 'n_interactions'] + int_cols
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, 0) for c in cols})

    if timed_out:
        with open(out_path + '.timeouts', 'w') as f:
            f.write('\n'.join(timed_out) + '\n')

    print(f'pocket {args.pocket}: {len(rows)}/{len(mols)} ok, {len(timed_out)} timeout, '
          f'{len(failed)} failed | protein {t_prot:.1f}s, total {time.time()-t0:.1f}s -> {out_path}')
    for name, err in failed[:5]:
        print(f'  [fail] {name}: {err}')


if __name__ == '__main__':
    sys.exit(main())
