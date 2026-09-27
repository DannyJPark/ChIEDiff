"""Build one comparison table across models from the gnina / smina / PoseBusters evaluation.

Reads each model's <dir>/per_molecule.csv (written by eval_aggregate_all.py) and emits:
    <out>/gnina_comparison.txt   human-readable tables
    <out>/gnina_comparison.csv   one row per model, machine-readable

Every number here is measured by THIS pipeline on the same footing: the same receptor files, the
same commands, and every generated molecule (no per-pocket subsampling). The Vina columns are the
numbers already stored in each model's sampling .pt -- they come from that model's own docking run,
so they are reported for continuity but are NOT on the same footing as the smina/gnina columns.

POPULATION. Pass --docked_only. Without it our rows cover every generated molecule while the
reference rows (targetdiff / kgdiff / pidiff) cover docking-successful molecules only, because
their .pt files were saved that way -- so the two are not counting the same thing and the extra
molecules on our side are exactly the fragmented ones. Measured on vina_fixed (9999 -> 8340):
smina score_only -9.696 -> -10.378, smina minimize -10.300 -> -10.703, PB-Valid(22) 30.0% ->
35.5%, MOLECULE pass 30.8% -> 36.4%, POSE pass 91.5% -> 96.7%. The reference rows do not move at
all. QED/SA are unaffected: eval_aggregate_all.py only fills those columns for docked molecules.
See scripts/mark_docked.py for the criterion and results/comparison/appendix/posebusters/pose_quality_summary.txt for
the same restriction applied to PoseCheck.

Aggregation follows the convention the SBDD literature uses, which is NOT uniform across columns:
  * affinity + SA + QED : mean/median over ALL molecules
  * PB-Valid + n_mols   : per-pocket rate, then mean/median over pockets
PB-Valid is reported on two check sets: 22 checks (posebusters 0.6.5, what we ran) and the 20-check
subset older versions used, because published PB-Valid numbers are on the 20-check basis and the
two are not comparable.

Usage:
    python scripts/build_gnina_comparison.py --dirs eval_gnina/targetdiff eval_out/kgdiff ... \
        --docked_only --out results/comparison
"""
import argparse
import csv
import os
from collections import defaultdict

import numpy as np

# The check set older posebusters versions (and therefore published PB-Valid numbers) used.
CHECKS_20 = [
    'mol_pred_loaded', 'mol_cond_loaded', 'sanitization', 'inchi_convertible', 'all_atoms_connected',
    'bond_lengths', 'bond_angles', 'internal_steric_clash', 'aromatic_ring_flatness',
    'double_bond_flatness', 'internal_energy', 'protein-ligand_maximum_distance',
    'minimum_distance_to_protein', 'minimum_distance_to_organic_cofactors',
    'minimum_distance_to_inorganic_cofactors', 'minimum_distance_to_waters',
    'volume_overlap_with_protein', 'volume_overlap_with_organic_cofactors',
    'volume_overlap_with_inorganic_cofactors', 'volume_overlap_with_waters',
]

# PoseBusters' checks split cleanly into two questions that a physics loss affects in OPPOSITE
# directions, and a single pass/fail rate hides that. Keep them apart.
CHECKS_MOL = [  # is the MOLECULE itself sane? (no protein involved)
    'sanitization', 'inchi_convertible', 'all_atoms_connected', 'no_radicals', 'bond_lengths',
    'bond_angles', 'internal_steric_clash', 'aromatic_ring_flatness',
    'non-aromatic_ring_non-flatness', 'double_bond_flatness', 'internal_energy',
]
CHECKS_POSE = [  # does it SIT in the pocket properly? (protein context)
    'protein-ligand_maximum_distance', 'minimum_distance_to_protein',
    'minimum_distance_to_organic_cofactors', 'minimum_distance_to_inorganic_cofactors',
    'minimum_distance_to_waters', 'volume_overlap_with_protein',
    'volume_overlap_with_organic_cofactors', 'volume_overlap_with_inorganic_cofactors',
    'volume_overlap_with_waters',
]


def stat(rows, col):
    v = [float(r[col]) for r in rows if r.get(col) not in (None, '')]
    if not v:
        return None, None, 0
    a = np.asarray(v)
    return float(a.mean()), float(np.median(a)), len(a)


def ring_stats(rows):
    """Ring pathology straight from the SMILES column -- no .pt needed."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    allene = Chem.MolFromSmarts('[#6]=[#6]=[#6]')
    nr, mx, n_macro, n_allene, n = [], [], 0, 0, 0
    for r in rows:
        smi = r.get('smiles')
        if not smi:
            continue
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        n += 1
        sizes = [len(x) for x in m.GetRingInfo().AtomRings()]
        nr.append(len(sizes))
        mx.append(max(sizes) if sizes else 0)
        if sizes and max(sizes) >= 8:
            n_macro += 1
        if m.HasSubstructMatch(allene):
            n_allene += 1
    if not n:
        return None
    return dict(rings=float(np.mean(nr)), maxring=float(np.mean(mx)),
                macro=100 * n_macro / n, allene=100 * n_allene / n, n=n)


def docked_set(model_dir):
    """-> set of docking-successful molecule names, or None if the model has no marker file.

    Written by scripts/mark_docked.py; the names are the SDF titles eval_export_sdf.py used, which
    are also the `name` column of per_molecule.csv and the `molecule` column of the PoseBusters
    CSVs, so one set filters every section.
    """
    p = os.path.join(model_dir, 'docked_names.txt')
    if not os.path.isfile(p):
        return None
    return {ln.strip() for ln in open(p) if ln.strip()}


def pb_rates(model_dir, keep=None):
    """-> (per-mol 22, per-pocket-mean 22, per-pocket-med 22, same for 20, n_eval, n_timeout)"""
    import glob
    pp22, pp20 = defaultdict(list), defaultdict(list)
    a22, a20 = [], []
    for f in sorted(glob.glob(os.path.join(model_dir, 'posebusters', 'pocket*.csv'))):
        pi = int(os.path.basename(f)[6:9])
        for row in csv.DictReader(open(f)):
            if not row.get('molecule'):
                continue
            if keep is not None and row['molecule'] not in keep:
                continue
            cols = {k: v for k, v in row.items()
                    if k not in ('file', 'molecule', 'position') and v != ''}
            if not cols:
                continue
            v22 = all(v == 'True' for v in cols.values())
            v20 = all(row[c] == 'True' for c in CHECKS_20 if c in row and row[c] != '')
            pp22[pi].append(v22); pp20[pi].append(v20); a22.append(v22); a20.append(v20)
    n_to = sum(sum(1 for _ in open(f))
               for f in glob.glob(os.path.join(model_dir, 'posebusters', '*.timeouts')))
    if not a22:
        return None
    p22 = np.array([np.mean(v) for v in pp22.values()])
    p20 = np.array([np.mean(v) for v in pp20.values()])
    # Timed-out molecules are NOT missing at random: they are the ones whose internal_energy check
    # could not embed a reference conformer in 900 s, i.e. chemically pathological structures that
    # would very likely have FAILED. Dropping them therefore flatters whichever model produces more
    # of them, and the rate differs ~30x between models. Report the conservative bound alongside.
    strict20 = 100 * sum(a20) / (len(a20) + n_to)
    strict22 = 100 * sum(a22) / (len(a22) + n_to)
    return dict(mol22=100 * np.mean(a22), pk22=100 * p22.mean(), pk22m=100 * np.median(p22),
                mol20=100 * np.mean(a20), pk20=100 * p20.mean(), pk20m=100 * np.median(p20),
                strict20=strict20, strict22=strict22,
                n_eval=len(a22), n_timeout=n_to, n_pockets=len(p22))


def pb_split(model_dir, keep=None):
    """Pass rate on the MOLECULE-intrinsic checks vs the POSE (protein-context) checks,
    plus the individual check failure rates."""
    import glob
    mol_ok = pose_ok = n = 0
    fails = defaultdict(int)
    for f in glob.glob(os.path.join(model_dir, 'posebusters', 'pocket*.csv')):
        for r in csv.DictReader(open(f)):
            if not r.get('molecule'):
                continue
            if keep is not None and r['molecule'] not in keep:
                continue
            n += 1
            if all(r.get(c) == 'True' for c in CHECKS_MOL if r.get(c) not in (None, '')):
                mol_ok += 1
            if all(r.get(c) == 'True' for c in CHECKS_POSE if r.get(c) not in (None, '')):
                pose_ok += 1
            for c in CHECKS_MOL + CHECKS_POSE:
                if r.get(c) == 'False':
                    fails[c] += 1
    if not n:
        return None
    return dict(mol=100 * mol_ok / n, pose=100 * pose_ok / n, n=n,
                top=[(k, 100 * v / n) for k, v in sorted(fails.items(), key=lambda x: -x[1])[:3]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dirs', nargs='+', required=True)
    ap.add_argument('--labels', nargs='+', default=None)
    ap.add_argument('--out', default='results/comparison/f2_pose_fidelity')
    # D5: not optional. A no-op for the reference models, which were saved docking-successful
    # only -- which is exactly why leaving it off compared our full generated set against their
    # filtered one. See posecheck-denominator-mismatch for the numbers it moves.
    ap.add_argument('--unfiltered', action='store_true',
                    help='DIAGNOSTIC ONLY: skip the docking-successful filter. Refuses to write '
                         'into results/comparison/ -- use results/diagnostics/.')
    args = ap.parse_args()
    args.docked_only = not args.unfiltered
    if args.unfiltered and 'results/comparison' in os.path.abspath(args.out):
        raise SystemExit('--unfiltered may not write into results/comparison/ -- '
                         'point --out at results/diagnostics/')
    labels = args.labels or [os.path.basename(os.path.normpath(d)) for d in args.dirs]
    os.makedirs(args.out, exist_ok=True)

    models = []
    for d, lab in zip(args.dirs, labels):
        pm = os.path.join(d, 'per_molecule.csv')
        if not os.path.exists(pm):
            print(f'[skip] {lab}: no per_molecule.csv in {d}')
            continue
        rows = list(csv.DictReader(open(pm)))
        keep = docked_set(d) if args.docked_only else None
        if keep is not None:
            before = len(rows)
            rows = [r for r in rows if r['name'] in keep]
            print(f'[docked_only] {lab}: {before} -> {len(rows)}')
        elif args.docked_only:
            print(f'[docked_only] {lab}: no docked_names.txt — *** row is still the full '
                  f'generated set ***')
        m = {'label': lab, 'dir': d, 'n_mols': len(rows),
             'n_pockets': len({r['pocket_idx'] for r in rows})}
        for c in ['vina_score_only', 'vina_dock', 'smina_score_affinity', 'smina_min_affinity',
                  'smina_min_rmsd', 'gnina_score_affinity', 'gnina_score_CNNscore',
                  'gnina_score_CNNaffinity', 'gnina_min_affinity', 'gnina_min_CNNscore',
                  'gnina_min_CNNaffinity', 'gnina_min_rmsd', 'qed', 'sa']:
            mean, med, n = stat(rows, c)
            m[c] = (mean, med, n)
        m['pb'] = pb_rates(d, keep)
        m['pbsplit'] = pb_split(d, keep)
        m['docked_only'] = args.docked_only
        m['rings'] = ring_stats(rows)
        models.append(m)

    L = []
    W = 20

    def hdr(title):
        L.append('')
        L.append('=' * 108)
        L.append(title)
        L.append('=' * 108)

    def table(title, cols, fmt='{:>9.3f}'):
        L.append('')
        L.append(title)
        L.append('-' * 108)
        L.append(f'{"Model":<{W}}' + ''.join(f'{c[1]:>18}' for c in cols))
        L.append(f'{"":<{W}}' + ''.join(f'{"avg":>9}{"med":>9}' for _ in cols))
        L.append('-' * 108)
        for m in models:
            line = f'{m["label"]:<{W}}'
            for key, _ in cols:
                mean, med, _n = m[key]
                line += (fmt.format(mean) + fmt.format(med)) if mean is not None else f'{"-":>9}{"-":>9}'
            L.append(line)

    hdr('gnina / smina / PoseBusters comparison  (measured by scripts/run_full_eval.sh)')
    L.append('')
    L.append('All affinities are kcal/mol, LOWER is better. CNNscore is a pose-quality probability')
    L.append('in [0,1] and CNNaffinity a pKd, so for BOTH of those HIGHER is better -- opposite to')
    L.append('every other column here.')
    L.append('')
    if args.docked_only:
        L.append('POPULATION: DOCKING-SUCCESSFUL MOLECULES ONLY. The reference .pt files (targetdiff,')
        L.append('kgdiff, pidiff) were saved that way, so this restriction is a no-op for them and')
        L.append('only trims our rows -- the molecules it removes are the fragmented ones, which is')
        L.append('a validity result reported separately, not something this table should absorb.')
        L.append('QED/SA are stored for docked molecules only, so those columns are unchanged.')
    else:
        L.append('*** POPULATION MISMATCH: our rows below cover EVERY generated molecule while the')
        L.append('    reference rows cover docking-successful molecules only (their .pt files hold')
        L.append('    nothing else). Do not compare rows. Re-run with --docked_only. ***')
    L.append('')
    L.append('A model appears here only if <dir>/per_molecule.csv exists (eval_aggregate_all.py).')
    if any('pidiff' in m['dir'].lower() for m in models):
        L.append('PIDiff comes from eval_out/pidiff, the CORRECTED export (99 pockets). Never point')
        L.append('this at eval_out/pidiff.BROKEN_renumbered_20260721 (92 pockets): that one renumbered')
        L.append('the flat .pt by order of appearance and paired 77 of 92 pockets with the WRONG')
        L.append('receptor, and it is the source of every PIDiff row in this file before 2026-07-24.')
        L.append('Its score_only/minimize columns were re-measured on the corrected export that day.')
    else:
        L.append('PIDiff is ABSENT. Its corrected export eval_out/pidiff needs score_only+minimize:')
        L.append('  DIR=eval_out/pidiff sbatch -p dell_cpu     -q cpu_qos  scripts/sbatch_eval_smina.sh')
        L.append('  DIR=eval_out/pidiff sbatch -p suma_rtx4090 -q base_qos scripts/sbatch_eval_gnina.sh')
        L.append('then eval_aggregate_all.py --pt .../PIDiff_vina_docked_complete.pt --dir eval_out/pidiff')
    L.append('')
    L.append(f'{"Model":<{W}}{"pockets":>9}{"molecules":>11}   source .pt')
    L.append('-' * 108)
    for m in models:
        L.append(f'{m["label"]:<{W}}{m["n_pockets"]:>9}{m["n_mols"]:>11}   {m["dir"]}')

    hdr('1. POSE AS GENERATED  (--score_only: the model\'s own pose, no coordinate moved)')
    table('smina / gnina affinity + gnina CNN scores', [
        ('smina_score_affinity', 'smina aff'),
        ('gnina_score_affinity', 'gnina aff'),
        ('gnina_score_CNNscore', 'gnina CNNscore'),
        ('gnina_score_CNNaffinity', 'gnina CNNaff'),
    ])
    L.append('')
    L.append('smina and gnina share the Vina 1.1.2 scoring function and the pose is fixed here, so')
    L.append('their affinity columns MUST agree to ~0.01. A gap means the two did not get the same')
    L.append('input -- treat it as a bug, not a result.')

    hdr('2. AFTER LOCAL MINIMIZATION  (--minimize: relax in the pocket, then score)')
    table('affinity + how far the pose had to move', [
        ('smina_min_affinity', 'smina aff'),
        ('gnina_min_affinity', 'gnina aff'),
        ('gnina_min_CNNscore', 'gnina CNNscore'),
        ('gnina_min_rmsd', 'move RMSD (A)'),
    ])
    L.append('')
    L.append('Read the affinity gain TOGETHER with the RMSD. A big improvement bought by a large')
    L.append('move means the generated pose was strained; it is not evidence of a good pose.')

    hdr('3. PoseBusters VALIDITY  (% of molecules passing EVERY check)')
    L.append('')
    L.append(f'{"Model":<{W}}{"22-check (0.6.5)":>28}{"20-check (older/published)":>34}{"evaluated":>11}{"timeout":>9}')
    L.append(f'{"":<{W}}{"per-mol":>10}{"pk avg":>9}{"pk med":>9}{"per-mol":>12}{"pk avg":>11}{"pk med":>11}')
    L.append('-' * 108)
    for m in models:
        pb = m['pb']
        if not pb:
            L.append(f'{m["label"]:<{W}}{"(no posebusters result)":>28}')
            continue
        L.append(f'{m["label"]:<{W}}'
                 f'{pb["mol22"]:>9.1f}%{pb["pk22"]:>8.1f}%{pb["pk22m"]:>8.1f}%'
                 f'{pb["mol20"]:>11.1f}%{pb["pk20"]:>10.1f}%{pb["pk20m"]:>10.1f}%'
                 f'{pb["n_eval"]:>11}{pb["n_timeout"]:>9}')
    L.append('')
    L.append('The 20-check column drops `no_radicals` and `non-aromatic_ring_non-flatness`, which')
    L.append('posebusters 0.6.5 added. Published PB-Valid numbers are on the 20-check basis; compare')
    L.append('against that column, never the 22-check one.')
    L.append('')
    L.append('TIMEOUTS ARE NOT MISSING AT RANDOM. A timed-out molecule is one whose internal_energy')
    L.append('check could not embed a reference conformer in 900 s -- a chemically pathological')
    L.append('structure that would very likely have FAILED. Excluding them flatters whichever model')
    L.append('produces more of them, and the rate is far from equal across models. Below: the same')
    L.append('20-check rate with every timeout counted INVALID (a lower bound, no selection bias).')
    L.append('')
    L.append(f'{"Model":<{W}}{"20-check, timeouts EXCLUDED":>30}{"20-check, timeouts INVALID":>30}{"delta":>10}')
    L.append('-' * 108)
    for m in models:
        pb = m['pb']
        if not pb:
            continue
        L.append(f'{m["label"]:<{W}}{pb["mol20"]:>29.1f}%{pb["strict20"]:>29.1f}%'
                 f'{pb["mol20"] - pb["strict20"]:>9.1f}p')

    hdr('3b. WHERE PoseBusters FAILS: the MOLECULE, or the POSE?')
    L.append('')
    L.append('A single PB-Valid number hides the most important thing here. Split the checks:')
    L.append('  MOLECULE  -- is the ligand itself sane? bond lengths/angles, ring flatness, radicals,')
    L.append('               internal clash, internal energy. The protein is not involved.')
    L.append('  POSE      -- does it sit in the pocket properly? min distance to protein/cofactors/')
    L.append('               waters, volume overlap. This is what a physics/affinity loss targets.')
    L.append('')
    L.append(f'{"Model":<{W}}{"MOLECULE pass":>15}{"POSE pass":>12}   worst individual checks')
    L.append('-' * 108)
    for m in models:
        s = m['pbsplit']
        if not s:
            continue
        top = '  '.join(f'{k}={v:.0f}%' for k, v in s['top'])
        L.append(f'{m["label"]:<{W}}{s["mol"]:>14.1f}%{s["pose"]:>11.1f}%   {top}')
    L.append('')
    L.append('Read this together: the physics/affinity-guided models beat TargetDiff on the POSE')
    L.append('checks -- their ligands really do sit in the pocket better -- and simultaneously lose')
    L.append('badly on the MOLECULE checks, overwhelmingly on bond_angles and bond_lengths. That is')
    L.append('consistent with what the loss actually constrains: the vdW/Vina term is a function of')
    L.append('LIGAND-PROTEIN distances only. Nothing in it penalises a distorted bond angle, so the')
    L.append('optimiser is free to buy intermolecular affinity by bending the ligand out of shape.')
    L.append('An intramolecular geometry term is the missing piece, not more physics on the contacts.')

    hdr('4. MOLECULAR PROPERTIES  (higher is better)')
    table('QED / SA', [('qed', 'QED'), ('sa', 'SA')])

    hdr('4b. RING PATHOLOGY  (from the SMILES; lower is better)')
    L.append('')
    L.append('Why this is here: macrocycles are exactly what makes PoseBusters\' internal_energy check')
    L.append('(ETKDG reference conformers) fail to converge, so this column EXPLAINS the timeout counts')
    L.append('above rather than being an independent finding. It is also a quality signal in its own')
    L.append('right -- an allene (C=C=C) or an 8+-membered ring in a de-novo ligand is almost always a')
    L.append('reconstruction artefact, not a design.')
    L.append('')
    L.append(f'{"Model":<{W}}{"rings/mol":>11}{"max ring":>10}{"macrocycle >=8":>16}{"allene C=C=C":>15}')
    L.append('-' * 108)
    for m in models:
        r = m['rings']
        if not r:
            continue
        L.append(f'{m["label"]:<{W}}{r["rings"]:>11.2f}{r["maxring"]:>10.2f}'
                 f'{r["macro"]:>15.1f}%{r["allene"]:>14.1f}%')

    hdr('5. PRE-EXISTING Vina NUMBERS  (from each model\'s own .pt -- NOT measured here)')
    table('Vina score_only / dock', [('vina_score_only', 'Vina score'), ('vina_dock', 'Vina dock')])
    L.append('')
    L.append('These are copied from whatever docking run produced each .pt -- not measured here.')
    L.append('The on-target protocol our models used DOES match the upstream targetdiff one (checked')
    L.append('against guanjq/targetdiff, 2026-07-20):')
    L.append('')
    L.append('                          box centre                    exhaustiveness')
    L.append('  upstream targetdiff     generated ligand\'s own bbox   16   (evaluate_diffusion.py')
    L.append('                          (pos.max+pos.min)/2, buffer 5       --exhaustiveness default)')
    L.append('  our models              same -- VinaDockingTask       16   (sbatch_dock_pocket.sh')
    L.append('                          .from_generated_mol()               EXHAUSTIVENESS default)')
    L.append('')
    L.append('(scripts/dock_generated_ligands.py DOES have a reference-ligand-centre box, but line 702')
    L.append('gates it behind `target_type != \'on_target\'` -- it is only for off-target selectivity')
    L.append('runs. On-target docking, which is what these columns are, takes the from_generated_mol')
    L.append('path. Note also that docking_vina.py\'s own `dock()` signature defaults to 8; nothing in')
    L.append('either pipeline uses that default, so it is a red herring.)')
    L.append('')
    L.append('What is still NOT verified: the baseline .pt files store no docking parameters, so the')
    L.append('exhaustiveness their runs actually used cannot be read back -- only assumed to be the')
    L.append('canonical 16. Prefer the smina/gnina sections above for cross-model claims regardless:')
    L.append('those are measured here, on every molecule, with identical commands and receptors.')
    L.append('')

    txt = '\n'.join(L)
    tpath = os.path.join(args.out, 'gnina_comparison.txt')
    with open(tpath, 'w') as f:
        f.write(txt + '\n')

    cpath = os.path.join(args.out, 'gnina_comparison.csv')
    cols = ['label', 'n_pockets', 'n_mols']
    metric_cols = ['smina_score_affinity', 'gnina_score_affinity', 'gnina_score_CNNscore',
                   'gnina_score_CNNaffinity', 'smina_min_affinity', 'gnina_min_affinity',
                   'gnina_min_CNNscore', 'gnina_min_CNNaffinity', 'gnina_min_rmsd',
                   'qed', 'sa', 'vina_score_only', 'vina_dock']
    with open(cpath, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols + [f'{c}_{s}' for c in metric_cols for s in ('avg', 'med')]
                   + ['pb_valid22_permol', 'pb_valid22_pocket_avg',
                      'pb_valid20_permol', 'pb_valid20_pocket_avg',
                      'pb_valid20_timeouts_invalid', 'pb_n_evaluated', 'pb_n_timeout'])
        for m in models:
            row = [m['label'], m['n_pockets'], m['n_mols']]
            for c in metric_cols:
                mean, med, _ = m[c]
                row += [f'{mean:.4f}' if mean is not None else '',
                        f'{med:.4f}' if med is not None else '']
            pb = m['pb']
            row += ([f'{pb["mol22"]:.2f}', f'{pb["pk22"]:.2f}', f'{pb["mol20"]:.2f}',
                     f'{pb["pk20"]:.2f}', f'{pb["strict20"]:.2f}',
                     pb['n_eval'], pb['n_timeout']] if pb else [''] * 7)
            w.writerow(row)

    print(txt)
    print(f'\n-> {tpath}')
    print(f'-> {cpath}')


if __name__ == '__main__':
    main()
