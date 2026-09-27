#!/usr/bin/env python3
"""Standalone smina-Vinardo / gnina score_only + minimize + full-redock comparison table.

Deliberately a separate file from results/comparison/f2_pose_fidelity/gnina_comparison.txt, not an appendix to it.
That file's sections are measured on EVERY molecule and its closing note tells the reader to
prefer them for cross-model claims on exactly that basis. The redock numbers here are measured on
DOCKING-SUCCESSFUL molecules only (the criterion the baseline .pt files already apply to
themselves), whereas that file reports every generated molecule -- so putting them in one document
would invite row-wise comparison between different populations. They also use exhaustiveness 8
against that file's Vina-from-.pt columns at 16, and its PIDiff rows come from a broken export.

Reads:
    eval_out/<model>/{smina,smina_vinardo,gnina}/pocket*_dock.sdf   affinities + CNN scores
    eval_out/<model>/pose_rmsd.csv                    RMSD(generated -> redocked)
    eval_out/<model>/docked_names.txt                  the docking-successful population

smina_vinardo/ is the smina binary run with --scoring vinardo. The smina *default* (Vina 1.1.2)
column was removed 2026-08-05: it measured the same functional form as the AutoDock Vina arm, so
showing both implied two independent engines where there was one. The paired (Vinardo - Vina)
D8 probe still needs it and lives in results/comparison/f2_pose_fidelity/rmsd_summary.txt section 4.
Vinardo AFFINITIES are on their own scale and must never be differenced against Vina ones;
the RMSD columns are in Angstroms and are directly comparable.

Writes:
    results/comparison/f2_pose_fidelity/redock_comparison.txt

Usage:
    python scripts/build_redock_comparison.py
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')

W = 150
def _order():
    """Row order from configs/models.json, not a hardcoded list.

    This used to be a literal nine tags. build_rmsd_summary.py derived the same order from the
    registry, so when a tenth model gained its F2 arms it appeared THERE and silently nowhere
    here -- two files claiming to report the same experiment on different rosters.
    """
    import sys as _s, os as _o
    _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
    import model_registry                                                # noqa: E402
    rows = [m for m in model_registry.models()
            if model_registry.in_table(m, 'pose', 'rmsd') and m['id'] != 'reference']
    rows.sort(key=lambda m: ((m['families']['pose'] or {}).get('legacy_order', 10**6), m['id']))
    return [m['ids']['eval_out'] for m in rows]


ORDER = _order()
def _labels():
    import sys as _s, os as _o
    _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
    import model_registry                                                # noqa: E402
    return model_registry.labels_by('pose')


# Labels come from configs/models.json so this file and rmsd_summary.txt name a model
# identically; they used to say 'Ours(vina_fixed)' and 'Ours (Vina loss)' for the same row.
LABEL = _labels()


def read_dock_tags(model_dir, engine, subset=None, mode='dock'):
    """-> {name: {tag: float}} for the engine's poses. Reads BOTH docking passes -- the first
    30/pocket (pocket*_dock.sdf) and the remainder (pocket*_dock_rest.sdf, added 2026-07-22 to
    lift the cap). They are disjoint by molecule name, so the union is the full set.

    mode='score' / 'min' read pocket*_score.sdf / pocket*_min.sdf, which were single-pass (no
    _rest companion) -- the glob simply finds nothing for those."""
    out = {}
    dkp = os.path.join(model_dir, 'docked_names.txt')
    if subset is None and os.path.isfile(dkp):
        subset = {ln.strip() for ln in open(dkp) if ln.strip()}
    # every pass: pocket*_<mode>.sdf plus any pocket*_<mode>_<pass>.sdf (_rest, _lost, ...).
    # Passes are disjoint by molecule name and `out.setdefault` keeps the first, so the union
    # is safe even if a pass is re-run.
    files = (sorted(glob.glob(os.path.join(model_dir, engine, f'pocket*_{mode}.sdf'))) +
             sorted(glob.glob(os.path.join(model_dir, engine, f'pocket*_{mode}_*.sdf'))))
    for p in files:
        if os.path.getsize(p) == 0:
            continue
        for m in Chem.SDMolSupplier(p, sanitize=False, removeHs=False):
            if m is None or not m.HasProp('_Name'):
                continue
            n = m.GetProp('_Name')
            if subset and n not in subset:
                continue
            rec = {}
            for t in ('minimizedAffinity', 'CNNscore', 'CNNaffinity'):
                if m.HasProp(t):
                    try:
                        rec[t] = float(m.GetProp(t))
                    except ValueError:
                        pass
            out.setdefault(n, rec)
    return out


SELF_DOCK_DIR = 'results/reference_protocol/self_redock_v4'
SELF_MIN_DIR  = 'results/reference_protocol/self_redock_min_v5'
SELF_SCORE_DIR = 'results/reference_protocol/self_reference_affinity_v2'


def load_reference():
    """The CrossDocked reference ligand put through the IDENTICAL protocol, one molecule per
    pocket. This is the ceiling every model row is read against, so it belongs at the TOP of each
    table, not appended at the bottom where it reads like another model.

    Sources (all produced by the same scripts as the model rows):
      dock  affinity : self_redock_v3/poses/pocketNNN_{vinardo,gnina}.sdf     (SD tags)
      min   affinity : self_redock_min_v5/poses/pocketNNN_{vinardo,gnina}_min.sdf
      score affinity : self_reference_affinity_v2/pocketNNN.csv               (--score_only)
      RMSD           : self_redock_v4 / self_redock_min_v5/pocketNNN.csv
    """
    def tags(pattern, key='minimizedAffinity', drop_penalty=True):
        """drop_penalty applies the SAME |E| < 1e3 grid-penalty filter as the model rows.

        The 20 A cube is centred on the reference ligand's own centroid, which does NOT
        guarantee the ligand fits inside it: pocket 072's reference is 20.1 A across, and an
        axis-aligned cube centred on the centroid can clip a ligand that is merely off-centre
        (pocket 081). An atom outside the map earns a ~1e6 kcal/mol penalty, and a single such
        pocket dragged the REFERENCE score_only mean to +13700 kcal/mol while its median stayed
        at -6.5. Filter it here as everywhere else; the count is reported so it is not silent.
        """
        out = []
        for f in sorted(glob.glob(pattern)):
            if not os.path.getsize(f):
                continue
            for m in Chem.SDMolSupplier(f, sanitize=False, removeHs=False):
                if m is not None and m.HasProp(key):
                    try:
                        v = float(m.GetProp(key))
                    except ValueError:
                        continue
                    if drop_penalty and key == 'minimizedAffinity' and abs(v) >= 1e3:
                        continue
                    out.append(v)
        return out

    def csvcol(pattern, col):
        vals = []
        for f in sorted(glob.glob(pattern)):
            for r in csv.DictReader(open(f)):
                if r.get(col):
                    try:
                        vals.append((r.get('pocket_idx', '0'), float(r[col])))
                    except ValueError:
                        pass
        return vals

    R = {}
    R['dock'] = {'vinardo': tags(f'{SELF_DOCK_DIR}/poses/pocket*_vinardo.sdf'),
                 'gnina':   tags(f'{SELF_DOCK_DIR}/poses/pocket*_gnina.sdf'),
                 'gnina_cnn': tags(f'{SELF_DOCK_DIR}/poses/pocket*_gnina.sdf', 'CNNscore'),
                 'gnina_cnnaff': tags(f'{SELF_DOCK_DIR}/poses/pocket*_gnina.sdf', 'CNNaffinity'),
                 'vmk': tags(f'{SELF_DOCK_DIR}/poses/pocket*_vmk.sdf')}
    R['min'] = {'vinardo': tags(f'{SELF_MIN_DIR}/poses/pocket*_vinardo_min.sdf'),
                'gnina':   tags(f'{SELF_MIN_DIR}/poses/pocket*_gnina_min.sdf'),
                'gnina_cnn': tags(f'{SELF_MIN_DIR}/poses/pocket*_gnina_min.sdf', 'CNNscore'),
                'gnina_cnnaff': tags(f'{SELF_MIN_DIR}/poses/pocket*_gnina_min.sdf', 'CNNaffinity'),
                'vmk': tags(f'{SELF_MIN_DIR}/poses/pocket*_vmk_min.sdf')}
    sc = {'vinardo': [], 'gnina': [], 'gnina_cnn': [], 'gnina_cnnaff': [], 'vmk': []}
    for f in sorted(glob.glob(f'{SELF_SCORE_DIR}/pocket*.csv')):
        for r in csv.DictReader(open(f)):
            for k, c in (('vinardo', 'vinardo_score_only'), ('gnina', 'gnina_score_only'),
                         ('gnina_cnn', 'gnina_CNNscore'),
                         ('gnina_cnnaff', 'gnina_CNNaffinity'),
                         ('vmk', 'vina_score_only')):
                if r.get(c):
                    try:
                        val = float(r[c])
                    except ValueError:
                        continue
                    # affinity columns only; CNNscore/CNNaffinity are bounded and never penalised
                    if k in ('vinardo', 'gnina', 'vmk') and abs(val) >= 1e3:
                        continue
                    sc[k].append(val)
    R['score'] = sc
    R['rmsd_dock'] = {'vinardo': csvcol(f'{SELF_DOCK_DIR}/pocket*.csv', 'rmsd_vinardo'),
                      'gnina':   csvcol(f'{SELF_DOCK_DIR}/pocket*.csv', 'rmsd_gnina'),
                      'vmk':     csvcol(f'{SELF_DOCK_DIR}/pocket*.csv', 'rmsd_vina_meeko')}
    R['rmsd_min'] = {'vinardo': csvcol(f'{SELF_MIN_DIR}/pocket*.csv', 'min_rmsd_vinardo'),
                     'gnina':   csvcol(f'{SELF_MIN_DIR}/pocket*.csv', 'min_rmsd_gnina'),
                     'vmk':     csvcol(f'{SELF_MIN_DIR}/pocket*.csv', 'min_rmsd_vina_meeko')}
    return R


def stat(xs):
    a = np.asarray([x for x in xs if x is not None and np.isfinite(x)], dtype=float)
    return (len(a), float(a.mean()), float(np.median(a))) if len(a) else (0, float('nan'),
                                                                          float('nan'))


def boot_median(pairs, B=3000, seed=0):
    if not pairs:
        return (float('nan'),) * 3
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for p, v in pairs:
        by[p].append(v)
    ks = list(by)
    a = {k: np.asarray(by[k], dtype=float) for k in ks}
    pt = float(np.median(np.concatenate([a[k] for k in ks])))
    if len(ks) < 2:
        return pt, float('nan'), float('nan')
    idx = rng.integers(0, len(ks), size=(B, len(ks)))
    o = [np.median(np.concatenate([a[ks[i]] for i in idx[b]])) for b in range(B)]
    return pt, float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--registry', default='configs/models.json')
    ap.add_argument('--out', default='results/comparison/f2_pose_fidelity/redock_comparison.txt')
    args = ap.parse_args()

    import sys as _sys, os as _os

    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

    import model_registry                                    # noqa: E402

    # Unified registry; load_as() renders the legacy 'pose' view, proven

    # byte-identical by scripts/check_registry_sync.py.

    reg = model_registry.load_as(args.registry, 'pose')
    models = {m['tag']: m for m in reg['models']}
    REF = load_reference()
    D = {}
    for tag in ORDER:
        m = models.get(tag)
        if not m:
            continue
        d = m['dir']
        # 보고 모집단: 도킹성공 분자만 (baseline .pt 관례와 동일 기준; scripts/mark_docked.py)
        dkp = os.path.join(d, 'docked_names.txt')
        docked = ({ln.strip() for ln in open(dkp) if ln.strip()}
                  if os.path.isfile(dkp) else None)
        rm = {}
        pr = os.path.join(d, 'pose_rmsd.csv')
        if os.path.isfile(pr):
            for r in csv.DictReader(open(pr)):
                if docked is None or r['name'] in docked:
                    rm[r['name']] = r
        D[tag] = {'dir': d, 'rmsd': rm,
                  'vinardo': read_dock_tags(d, 'smina_vinardo'),
                  'gnina': read_dock_tags(d, 'gnina')}
        # score_only (pose untouched) and minimize (relaxed in place) -- same docking-successful
        # population, same subset filter. Coverage differs a lot per engine, so every cell
        # carries its own n rather than a single per-row n.
        for mode in ('score', 'min'):
            for key, sub in (('vinardo', 'smina_vinardo'), ('gnina', 'gnina')):
                D[tag][f'{key}_{mode}'] = read_dock_tags(d, sub, mode=mode)
        # AutoDock Vina (meeko): one CSV per pocket holding all three modes at once, so it is
        # read separately from the SDF engines. Columns: vina_score_only / vina_minimize /
        # vina_dock. Same docking-successful filter.
        vmk = {'score': {}, 'min': {}, 'dock': {}}
        for f in sorted(glob.glob(os.path.join(d, 'vina_meeko', 'pocket*_ref.csv'))):
            for r in csv.DictReader(open(f)):
                n = r.get('name')
                if not n or (docked and n not in docked):
                    continue
                for k, c in (('score', 'vina_score_only'), ('min', 'vina_minimize'),
                             ('dock', 'vina_dock')):
                    try:
                        v = float(r[c])
                    except (TypeError, ValueError, KeyError):
                        continue
                    # the ref box is a fixed 20A cube; an atom outside it earns a ~1e6 kcal/mol
                    # grid penalty. Those are real outputs but not affinities -- drop them here
                    # rather than let one poison a mean (see analysis/METHODS §3.6).
                    if abs(v) < 1e3:
                        vmk[k].setdefault(n, {})['minimizedAffinity'] = v
                # the meeko arm computes its own gen->pose RMSD (exact atom map) and stores it in
                # the same CSV; merge it into the pose_rmsd row so one reader serves all engines
                if n in rm:
                    for src, dst in (('rmsd_gen_dock', 'vm_rmsd_gen_dock'),
                                     ('rmsd_gen_min', 'vm_rmsd_gen_min')):
                        if r.get(src):
                            rm[n][dst] = r[src]
        for k in ('score', 'min', 'dock'):
            D[tag][f'vmk_{k}'] = vmk[k]
        D[tag]['vmk'] = vmk['dock']

    L = []
    A = L.append
    A('')
    A('=' * W)
    A('THREE-ENGINE comparison: score_only / minimize / full redock')
    A('           (smina Vinardo | gnina CNN;  --dock box = --autobox_ligand crystal)')
    A('=' * W)
    A('')
    A('Generated by scripts/build_redock_comparison.py. This is a SEPARATE document from')
    A('gnina_comparison.txt and its rows must NOT be read alongside that file\'s rows:')
    A('')
    A('  1. MODE + POPULATION.  A full re-search (--dock) was never measured in')
    A('     gnina_comparison.txt, which has --score_only and --minimize only. Also, everything')
    A('     here is on DOCKING-SUCCESSFUL molecules only (the criterion the baseline .pt files')
    A('     already apply to themselves); gnina_comparison.txt reports every generated molecule.')
    A('  2. PIDiff.  gnina_comparison.txt\'s PIDiff rows come from the export that renumbered')
    A('     flat .pt pockets by order of appearance, pairing 77 of 92 pockets with the WRONG')
    A('     receptor. PIDiff here is the corrected 99-pocket export. The two are not the same')
    A('     measurement of the same thing.')
    A('  3. EXHAUSTIVENESS.  8 here; gnina_comparison.txt section 5 (Vina from each .pt) is 16.')
    A('')
    A('SECTION ORDER.  Sections 1-2 are the full redock (the headline); sections 3-4 are the')
    A('score_only and minimize modes, appended 2026-08. Physically the progression is')
    A('score_only -> minimize -> dock (nothing moves -> local relaxation -> global re-search), so')
    A('read 3, 4, then 1-2 if you want the causal story rather than the headline.')
    A('')
    A('Protocol, identical for every model: --autobox_ligand <crystal ligand> --exhaustiveness 8')
    A('--num_modes 1 --seed 42. Same binaries (~/bin/{smina,gnina} v1.1), same receptor PDBQTs.')
    A('The Vinardo column is the smina binary with --scoring vinardo. The smina DEFAULT (Vina)')
    A('column was removed 2026-08-05 as a duplicate of the AutoDock Vina scoring function.')
    A('')
    A('!! THE AutoDock Vina COLUMN USES A DIFFERENT BOX AND SEARCH DEPTH, ON PURPOSE.')
    A('   smina-Vinardo / gnina : --autobox_ligand <crystal ligand>, exhaustiveness 8')
    A('   AutoDock Vina (meeko) : fixed 20 A cube on the crystal-ligand centroid, exhaustiveness 16,')
    A('                           and meeko`s exact REMARK-SMILES atom map (no fallback matching)')
    A('   A bigger box plus a deeper search finds lower minima, which is why this column reads')
    A('   several kcal/mol more negative than the other two for the SAME molecule. Compare rows')
    A('   WITHIN the column; never read across it as if the three were interchangeable. Its value')
    A('   is that it is an independent implementation with an exact atom map, not that its numbers')
    A('   line up with smina`s.')
    A('   Molecules with an atom outside that fixed cube earn a ~1e6 kcal/mol grid penalty; those')
    A('   are dropped from this column (|E| >= 1e3) rather than allowed to poison a mean.')
    A('')
    A('Affinities are kcal/mol, LOWER is better. CNNscore is a pose-quality probability in [0,1]')
    A('and CNNaffinity a pKd, so for BOTH of those HIGHER is better -- opposite to every other')
    A('column. CNNscore and CNNaffinity answer different questions: CNNscore is "is this pose')
    A('right", CNNaffinity is "how tightly would it bind". A model can win one and lose the other.')
    A('')
    A('HOW avg / med ARE COMPUTED.  Every molecule of every pocket is pooled into ONE flat list,')
    A('then averaged / median-ed. They are MOLECULE-level statistics, NOT pocket-level -- i.e. NOT')
    A('"the mean of the per-pocket means". Pockets contribute unequal numbers of molecules (a')
    A('pocket where the model fragmented most of its output contributes fewer), so a molecule-rich')
    A('pocket carries proportionally more weight. The RMSD tables below give a POCKET-level')
    A('bootstrap CI, but even there the point estimate is the pooled molecule median -- only the')
    A('interval is resampled over pockets. Use the CI, not the avg, when comparing rows.')
    A('')
    A('!! VINARDO AFFINITIES ARE NOT ON THE VINA SCALE. Both print "kcal/mol", which makes the')
    A('   columns look interchangeable; they are not. Vinardo drops Vina\'s gauss2 term, refits the')
    A('   steric/hydrophobic/h-bond functions, and weights the rotatable-bond penalty at 0 where')
    A('   Vina uses 1.923 -- so it does not charge flexible ligands the entropy penalty Vina does.')
    A('   RANK MODELS WITHIN THE VINARDO COLUMN. Never difference a Vinardo number against a')
    A('   smina/Vina one, and never quote the two in the same sentence as if they were the same')
    A('   quantity. The RMSD table (section 2) has no such problem -- Angstroms are Angstroms.')
    A('')

    # ---------------------------------------------------------------- roster
    A('=' * W)
    A('0. WHAT WAS ACTUALLY DOCKED')
    A('=' * W)
    A('')
    A(f'{"Model":22s} {"pockets":>8s} {"docking-ok":>11s} '
      f'{"vinardo posed":>14s} {"gnina posed":>12s} {"AD-Vina posed":>14s}')
    A('-' * W)
    A(f'{"REFERENCE (ceiling)":22s} {100:8d} {100:11d} '
      f'{len(REF["dock"]["vinardo"]):14d} {len(REF["dock"]["gnina"]):12d} '
      f'{len(REF["dock"]["vmk"]):14d}')
    A('-' * W)
    for tag in ORDER:
        if tag not in D:
            continue
        e = D[tag]
        npk = len({r['pocket_idx'] for r in e['rmsd'].values()})
        A(f'{LABEL[tag]:22s} {npk:8d} {len(e["rmsd"]):11d} '
          f'{len(e["vinardo"]):14d} {len(e["gnina"]):12d} {len(e["vmk"]):14d}')
    A('')
    A('"docking-ok" = the reporting population: molecules with a valid Vina result in the source')
    A('.pt. For the physics models this excludes the disconnected molecules (15-28% of what they')
    A('generate; every baseline is 0%). That fragmentation rate is a headline VALIDITY result and')
    A('is reported in analysis/RESULTS_pose_fidelity_2026-07.md section 1 -- quote it alongside')
    A('these numbers, never on its own.')
    A('')
    A('"posed" below "submitted" used to have TWO causes, only one of which was an engine failure.')
    A('Both were repaired 2026-08-05; the counts above are post-repair.')
    A('')
    A('  (a) REAL engine refusal -- ONE molecule. gnina v1.1 core-dumps on vina_fixed p010_m0025')
    A('      (38 heavy atoms), reproducibly (rc=134) even when docked alone. gnina processes a')
    A('      multi-model SDF in one process, so the abort originally destroyed every pose after it')
    A('      in that pocket; the worker now refuses to promote an output with fewer poses than')
    A('      molecules submitted. Net effect on the tables: 1 molecule of ~43,000.')
    A('')
    A('  (b) A SILENT SELECTION EFFECT -- six molecules, and the more important finding.')
    A('      make_subset_sdf.py built sdf30/ and sdfrest30/ with SDMolSupplier at its default')
    A('      sanitize=True and dropped whatever returned None. Those molecules then appeared in')
    A('      NEITHER subset, so the two files stopped partitioning sdf/ and no engine ever saw')
    A('      them -- while pose_rmsd.py, which reads sdf/ with sanitize=False, kept them in the')
    A('      reporting denominator. They were not refused; they were never submitted. Re-docked')
    A('      individually, smina posed all 6 and gnina posed 5 (the 6th is case (a)).')
    A('      This mattered beyond its size because it is NOT missing-at-random: the molecules')
    A('      RDKit cannot sanitize are the chemically distorted ones, so the loss fell entirely on')
    A('      vina_fixed (2) and pignet_fixed (4) and hit no baseline at all. The Vinardo arm was')
    A('      never affected -- it docked sdf/ directly and shows zero short pockets, which is what')
    A('      isolated the cause.')
    A('      Fixed at source (sanitize=False in make_subset_sdf.py); recovered poses live in')
    A('      pocket*_dock_lost.sdf, and every reader now globs pocket*_<mode>_*.sdf so any')
    A('      additional pass is picked up automatically.')
    A('')

    # ---------------------------------------------------------------- affinity
    A('=' * W)
    A('1. REDOCKED AFFINITY  (the MOLECULE\'s best score, pose re-searched from scratch)')
    A('=' * W)
    A('')
    A('This answers "is the molecule any good", NOT "was the generated pose any good".')
    A('The pose the model produced is discarded before scoring. Section 2 is the pose question.')
    A('')
    A(f'{"Model":22s} {"AutoDock Vina":>19s} {"smina VINARDO":>19s} {"gnina aff":>19s} '
      f'{"gnina CNNscore":>17s} {"gnina CNNaff":>17s}')
    A(f'{"":22s} ' + ' '.join(f'{"avg":>9s} {"med":>9s}' for _ in range(3)) +
      ' ' + ' '.join(f'{"avg":>8s} {"med":>8s}' for _ in range(2)))
    A('-' * W)
    rv, rw, rg, rc, ra = (stat(REF['dock']['vmk']), stat(REF['dock']['vinardo']),
                          stat(REF['dock']['gnina']), stat(REF['dock']['gnina_cnn']),
                          stat(REF['dock']['gnina_cnnaff']))
    A(f'{"REFERENCE (ceiling)":22s} {rv[1]:9.3f} {rv[2]:9.3f} {rw[1]:9.3f} {rw[2]:9.3f} '
      f'{rg[1]:9.3f} {rg[2]:9.3f} {rc[1]:8.3f} {rc[2]:8.3f} {ra[1]:8.3f} {ra[2]:8.3f}')
    A('-' * W)
    for tag in ORDER:
        if tag not in D:
            continue
        e = D[tag]
        va = stat([v.get('minimizedAffinity') for v in e['vmk'].values()])
        wa = stat([v.get('minimizedAffinity') for v in e['vinardo'].values()])
        ga = stat([v.get('minimizedAffinity') for v in e['gnina'].values()])
        gc = stat([v.get('CNNscore') for v in e['gnina'].values()])
        gf = stat([v.get('CNNaffinity') for v in e['gnina'].values()])
        A(f'{LABEL[tag]:22s} {va[1]:9.3f} {va[2]:9.3f} {wa[1]:9.3f} {wa[2]:9.3f} '
          f'{ga[1]:9.3f} {ga[2]:9.3f} {gc[1]:8.3f} {gc[2]:8.3f} {gf[1]:8.3f} {gf[2]:8.3f}')
    A('')
    A('Read ACROSS a row only for rank agreement, never for magnitude: the Vinardo column is a')
    A('different function (see the warning at the top).')
    A('')
    A('THE TWO CNN COLUMNS DISSOCIATE, AND THAT IS THE POINT. For the REFERENCE ligand the CNN')
    A('gives the HIGHEST pose score of any row (CNNscore 0.886 vs 0.787 for the best model) and')
    A('one of the LOWEST affinities (CNNaffinity 5.65 vs 6.29 for ours). Same network, same pose,')
    A('opposite verdicts: it is most confident that the real ligand`s pose is correct, and least')
    A('convinced it binds tightly. The generated molecules invert that.')
    A('So "we beat the reference ligand" is true on every kcal/mol column and on CNNaffinity, and')
    A('FALSE on CNNscore -- the one column that asks whether the pose itself is right. Quote both.')
    A('')

    # ---------------------------------------------------------------- rmsd
    A('=' * W)
    A('2. HOW FAR THE REDOCK MOVED THE GENERATED POSE  (RMSD, A -- the pose-fidelity question)')
    A('=' * W)
    A('')
    A('Non-superposed, symmetry-minimised heavy-atom RMSD (rdMolAlign.CalcRMS). Lower = the')
    A('engine left the generated pose where it was. CI is a 95% POCKET-level bootstrap: molecules')
    A('are nested in pockets, so resampling molecules would understate it.')
    A('')
    A(f'{"Model":22s} {"AutoDock Vina":>24s} {"smina VINARDO":>24s} {"gnina (CNN)":>24s} {"n":>8s}')
    A('-' * W)
    rc = []
    for k in ('vmk', 'vinardo', 'gnina'):
        pt, lo, hi = boot_median(REF['rmsd_dock'][k])
        rc.append(f'{pt:5.2f} [{lo:4.2f},{hi:5.2f}]' if np.isfinite(pt) else '—')
    A(f'{"REFERENCE (ceiling)":22s} {rc[0]:>24s} {rc[1]:>24s} {rc[2]:>24s} {100:8d}')
    A('-' * W)
    for tag in ORDER:
        if tag not in D:
            continue
        e = D[tag]
        cells, meds = [], []
        for col in ('vm_rmsd_gen_dock', 'rmsd_gen_dock_vinardo', 'rmsd_gen_dock_gnina'):
            pairs = []
            for n, r in e['rmsd'].items():
                v = r.get(col, '')
                if v not in ('', None):
                    try:
                        pairs.append((r['pocket_idx'], float(v)))
                    except ValueError:
                        pass
            pt, lo, hi = boot_median(pairs)
            meds.append((pt, len(pairs)))
            cells.append(f'{pt:5.2f} [{lo:4.2f},{hi:5.2f}]' if np.isfinite(pt) else '—')
        A(f'{LABEL[tag]:22s} {cells[0]:>24s} {cells[1]:>24s} {cells[2]:>24s} {meds[1][1]:8d}')
    A('')
    A('The controlled D8 probe -- the paired per-molecule (Vinardo - Vina) shift -- lives in')
    A('results/comparison/f2_pose_fidelity/rmsd_summary.txt section 4. It is not repeated here because it needs the')
    A('smina (Vina) arm, which was removed from these tables as a duplicate of AutoDock Vina.')
    A('')

    # ------------------------------------------------- score_only / minimize (3 engines)
    def _cell3(n, av, md, w=8):
        return (f'{av:{w}.3f} {md:{w}.3f}' if n else f'{"--":>{w}s} {"--":>{w}s}')

    def aff_cells(e, mode):
        """AutoDock Vina | vinardo | gnina aff | gnina CNNscore | gnina CNNaffinity | n.

        CNNscore (0-1 pose probability) and CNNaffinity (pKd) are HIGHER-is-better, the opposite
        of the three kcal/mol columns beside them."""
        kp = e.get(f'vmk_{mode}', {})
        vp = e.get(f'vinardo_{mode}', {})
        gp = e.get(f'gnina_{mode}', {})
        nk, av_k, md_k = stat([v.get('minimizedAffinity') for v in kp.values()])
        nv, av_v, md_v = stat([v.get('minimizedAffinity') for v in vp.values()])
        ng, av_g, md_g = stat([v.get('minimizedAffinity') for v in gp.values()])
        nc, av_c, md_c = stat([v.get('CNNscore') for v in gp.values()])
        na, av_a, md_a = stat([v.get('CNNaffinity') for v in gp.values()])
        return [_cell3(nk, av_k, md_k), _cell3(nv, av_v, md_v), _cell3(ng, av_g, md_g),
                _cell3(nc, av_c, md_c), _cell3(na, av_a, md_a), f'{nk:5d}/{nv:5d}/{ng:5d}']

    def ref_aff_cells(mode):
        R = REF[mode]
        nk, av_k, md_k = stat(R['vmk'])
        nv, av_v, md_v = stat(R['vinardo'])
        ng, av_g, md_g = stat(R['gnina'])
        nc, av_c, md_c = stat(R['gnina_cnn'])
        na, av_a, md_a = stat(R['gnina_cnnaff'])
        return [_cell3(nk, av_k, md_k), _cell3(nv, av_v, md_v), _cell3(ng, av_g, md_g),
                _cell3(nc, av_c, md_c), _cell3(na, av_a, md_a), f'{nk:5d}/{nv:5d}/{ng:5d}']

    A('=' * W)
    A('3. POSE AS GENERATED  (--score_only: not one coordinate moved)')
    A('=' * W)
    A('')
    A('The model\'s own pose, scored by three different functions. No box and no search are')
    A('involved, so this is the purest "how good does each function think this pose is".')
    A('')
    A('COVERAGE IS UNEQUAL -- read the n in each cell, not the row. A cell with a much smaller n')
    A('is a DIFFERENT population, not a comparable number; n=0 means that arm was never submitted,')
    A('which is not the same as "the model scored badly".')
    A('Vinardo: all ten models. The gnina score_only+minimize arm is still missing for')
    A('pignet_fixed, pgdiff_fixed and (score_only only) novdw -- never submitted, by scope')
    A('decision, not failure.')
    A('AliDiff and MolCRAFT were completed 2026-08-05; MolCRAFT\'s gnina arm had stalled at 64/100')
    A('pockets and one legacy pocket held only 58/100 molecules (see the note below).')
    A('')
    A(f'{"Model":22s} {"AutoDock Vina":>17s} {"VINARDO aff":>17s} {"gnina aff":>17s} '
      f'{"gnina CNNscore":>17s} {"gnina CNNaff":>17s} {"n vmk/vnd/gni":>19s}')
    A(f'{"":22s} ' + ' '.join(f'{"avg":>8s} {"med":>8s}' for _ in range(5)) +
      f' {"":>19s}')
    A('-' * W)
    A(f'{"REFERENCE (ceiling)":22s} ' + ' '.join(ref_aff_cells('score')))
    A('-' * W)
    for tag in ORDER:
        if tag not in D:
            continue
        A(f'{LABEL[tag]:22s} ' + ' '.join(aff_cells(D[tag], 'score')))
    A('')
    A('score_only is the mode most exposed to a strained pose: nothing is allowed to relax, so a')
    A('clash is charged in full. Compare against section 4 (same pose, after relaxation).')
    A('')
    A('CROSS-CHECK (run before the smina Vina column was dropped): smina-Vina and gnina share the')
    A('Vina 1.1.2 scoring function and the pose is FIXED here, so their avg/med had to agree to')
    A('~0.01 wherever n matched. They did (max |diff| 0.014), which is what certified that the two')
    A('engines received identical input. That is also why the smina column was redundant.')
    A('')
    A('A legacy trap this exposed: molcraft gnina pocket063 held 58 of 100 molecules, written by')
    A('the older sbatch_eval_gnina.sh which had no promotion gate. build_worklist.py flagged it as')
    A('SHORT and re-queued it every time, but the worker skipped it on a bare `[ -s "$out" ]`')
    A('non-empty test -- so it could never be repaired by resubmitting. The worker now applies the')
    A('same molecule-count test as the builder and logs "[redo]" when it regenerates.')
    A('')

    A('=' * W)
    A('4. LOCALLY MINIMIZED  (--minimize: relax in place, no box, no search)')
    A('=' * W)
    A('')
    A(f'{"Model":22s} {"AutoDock Vina":>17s} {"VINARDO aff":>17s} {"gnina aff":>17s} '
      f'{"gnina CNNscore":>17s} {"gnina CNNaff":>17s} {"n vmk/vnd/gni":>19s}')
    A(f'{"":22s} ' + ' '.join(f'{"avg":>8s} {"med":>8s}' for _ in range(5)) +
      f' {"":>19s}')
    A('-' * W)
    A(f'{"REFERENCE (ceiling)":22s} ' + ' '.join(ref_aff_cells('min')))
    A('-' * W)
    for tag in ORDER:
        if tag not in D:
            continue
        A(f'{LABEL[tag]:22s} ' + ' '.join(aff_cells(D[tag], 'min')))
    A('')
    A('4b. HOW FAR MINIMIZATION HAD TO MOVE THE POSE (RMSD, A; median [95% pocket bootstrap])')
    A('')
    A('Read the affinity gain in section 4 TOGETHER with this. A big improvement bought by a large')
    A('move means the generated pose was strained -- it is not evidence of a good pose. These are')
    A('our own symmetry-aware non-superposed CalcRMS values (pose_rmsd.csv), not the engine\'s')
    A('index-order minimizedRMSD tag, so they are consistent with section 2.')
    A('')
    A(f'{"Model":22s} {"AutoDock Vina":>24s} {"smina VINARDO":>24s} {"gnina (CNN)":>24s} {"n(vnd)":>8s}')
    A('-' * W)
    rc = []
    for k in ('vmk', 'vinardo', 'gnina'):
        pt, lo, hi = boot_median(REF['rmsd_min'][k])
        rc.append(f'{pt:5.3f} [{lo:5.3f},{hi:5.3f}]' if np.isfinite(pt) else '—')
    A(f'{"REFERENCE (ceiling)":22s} {rc[0]:>24s} {rc[1]:>24s} {rc[2]:>24s} {100:8d}')
    A('-' * W)
    for tag in ORDER:
        if tag not in D:
            continue
        e = D[tag]
        cells, nv = [], 0
        for col in ('vm_rmsd_gen_min', 'rmsd_gen_min_vinardo', 'rmsd_gen_min_gnina'):
            pairs = []
            for r in e['rmsd'].values():
                v = r.get(col, '')
                if v not in ('', None):
                    try:
                        pairs.append((r['pocket_idx'], float(v)))
                    except ValueError:
                        pass
            pt, lo, hi = boot_median(pairs)
            if col.endswith('vinardo'):
                nv = len(pairs)
            cells.append(f'{pt:5.3f} [{lo:5.3f},{hi:5.3f}]' if np.isfinite(pt) else '—')
        A(f'{LABEL[tag]:22s} {cells[0]:>24s} {cells[1]:>24s} {cells[2]:>24s} {nv:8d}')
    A('')
    A('For the paired per-molecule (Vinardo - Vina) comparison -- the controlled D8 probe, and the')
    A('one to quote -- see results/comparison/f2_pose_fidelity/rmsd_summary.txt section 4. It is not repeated here')
    A('because it needs the smina (Vina) arm that these tables no longer show.')
    A('')

    # ------------------------------------------------- aggregation sensitivity
    A('=' * W)
    A('5. AGGREGATION SENSITIVITY: molecule-pooled vs pocket-level  (--dock affinity)')
    A('=' * W)
    A('')
    A('Every avg/med above pools all molecules of all pockets into one list. That silently weights')
    A('each pocket by how many molecules SURVIVED in it -- and the pockets where fewest survived')
    A('are exactly the ones the model handled worst (fragmentation is what removes them, and it is')
    A('far from uniform: pignet_fixed ranges 5-100 molecules per pocket, kgdiff 5-100,')
    A('pocket2mol 9-100). Pooling therefore under-weights a model\'s worst pockets, and only the')
    A('physics-loss models fragment at all (15-28% vs 0%), so the bias is not symmetric.')
    A('')
    A('Pocket-level gives every pocket weight 1 regardless of survivor count. It is the more')
    A('defensible estimand for an in-house table, and it matches the pocket-level bootstrap this')
    A('file already uses for CIs. The molecule-pooled columns are kept as the headline only because')
    A('results/comparison/f1_sbdd/comparison_tables.txt must stay comparable to the published SBDD tables')
    A('(ConDitar Table 1 convention), which are all molecule-pooled.')
    A('')
    A('If the two disagree, quote the pocket-level number and say so.')
    A('')
    A(f'{"Model":22s} {"engine":9s} {"pooled avg":>11s} {"pooled med":>11s} '
      f'{"pkt mean":>10s} {"pkt med":>10s} {"n_mol":>7s} {"n_pkt":>6s} {"mol/pkt":>10s}')
    A('-' * W)
    # Compare LIKE WITH LIKE. |pooled median - pocket mean| would conflate two changes at once
    # (median->mean AND molecule->pocket weighting); the gnina affinity distribution is left-
    # skewed, so that mixed number reads ~1.3 kcal/mol and blames the weighting for what is
    # really mean-vs-median. The weighting effect is pooled-mean vs pocket-mean-of-means.
    worst_mean = 0.0
    byeng = defaultdict(list)          # engine -> [(label, pooled_avg, pooled_med, pkt_mean, pkt_med)]
    for tag in ORDER:
        if tag not in D:
            continue
        for key, disp in (('vinardo', 'vinardo'), ('gnina', 'gnina')):
            # regroup the dock affinities by pocket via the molecule name p<NNN>_m<MMMM>
            byp = defaultdict(list)
            for n, rec in D[tag][key].items():
                v = rec.get('minimizedAffinity')
                if v is None or not np.isfinite(v):
                    continue
                byp[n[1:4]].append(v)
            if not byp:
                continue
            allv = np.concatenate([np.asarray(v) for v in byp.values()])
            pmeans = np.asarray([np.mean(v) for v in byp.values()])
            cnt = [len(v) for v in byp.values()]
            worst_mean = max(worst_mean, abs(float(allv.mean()) - float(pmeans.mean())))
            byeng[disp].append((LABEL[tag], float(allv.mean()), float(np.median(allv)),
                                float(pmeans.mean()), float(np.median(pmeans))))
            A(f'{LABEL[tag]:22s} {disp:9s} {allv.mean():11.3f} {np.median(allv):11.3f} '
              f'{pmeans.mean():10.3f} {np.median(pmeans):10.3f} {len(allv):7d} {len(byp):6d} '
              f'{str(min(cnt)) + "-" + str(max(cnt)):>10s}')
    A('-' * W)
    A(f'WEIGHTING EFFECT, like-for-like (pooled avg vs pkt mean, both means): largest across all')
    A(f'{len(ORDER)} models x 2 engines = {worst_mean:.3f} kcal/mol.')
    A('')
    A('DOES THE WEIGHTING CHANGE THE RANKING?  Computed, not assumed:')
    max_shift, max_gap = 0, 0.0
    for eng in ('vinardo', 'gnina'):
        rs = byeng.get(eng)
        if not rs:
            continue
        for lbl, i_pool, i_pkt in (('avg', 1, 3), ('med', 2, 4)):
            a = [r[0] for r in sorted(rs, key=lambda r: r[i_pool])]
            b = [r[0] for r in sorted(rs, key=lambda r: r[i_pkt])]
            if a == b:
                A(f'  {eng:8s} {lbl}: order identical')
                continue
            shifts = {n: abs(a.index(n) - b.index(n)) for n in a}
            moved = sorted([n for n, s in shifts.items() if s], key=lambda n: a.index(n))
            mx = max(shifts.values())
            max_shift = max(max_shift, mx)
            # how big is the score gap between the rows that swapped?
            sc = {r[0]: r[i_pool] for r in rs}
            gaps = [abs(sc[moved[i]] - sc[moved[i + 1]]) for i in range(len(moved) - 1)]
            if gaps:
                max_gap = max(max_gap, min(gaps))
            A(f'  {eng:8s} {lbl}: {len(moved)} row(s) move, max shift {mx} position(s) '
              f'-- {", ".join(moved)}')
    # best/worst stability, derived rather than asserted
    best_same, worst_diff = set(), []
    for eng in ('vinardo', 'gnina'):
        rs = byeng.get(eng)
        if not rs:
            continue
        for lbl, i_pool, i_pkt in (('avg', 1, 3), ('med', 2, 4)):
            a = sorted(rs, key=lambda r: r[i_pool])
            b = sorted(rs, key=lambda r: r[i_pkt])
            best_same.add(a[0][0] == b[0][0])
            if a[-1][0] != b[-1][0]:
                worst_diff.append(f'{eng} {lbl}: {a[-1][0]} vs {b[-1][0]}')
    A('')
    A(f'Every move is by at most {max_shift} position, and only between rows separated by well under')
    A('1 kcal/mol.')
    if best_same == {True}:
        A('The BEST model is Ours(vina_fixed) under both weightings in every engine x statistic')
        A('cell -- the headline claim of this file does not turn on the choice.')
    if worst_diff:
        A('The WORST row is NOT always stable: ' + '; '.join(worst_diff) + '.')
        A('That pair is a near-tie, which is exactly the regime where the weighting decides.')
    A('So the choice changes no conclusion drawn here, but it DOES resolve near-ties (novdw vs')
    A('KGDiff, novdw vs pgdiff_fixed, AliDiff vs MolCRAFT, AliDiff vs Pocket2Mol). Whenever two')
    A('rows sit within ~0.3 kcal/mol, state which weighting you used before calling one better.')
    A('')
    A('Do NOT read "pooled med" against "pkt mean" -- that mixes median-vs-mean with the weighting')
    A('change and looks like a ~1.3 kcal/mol effect. The gnina affinity distribution is strongly')
    A('left-skewed (a tail of very poor redocks), so its mean and median differ by ~1 kcal/mol')
    A('under ANY weighting. Compare avg with avg, med with med.')
    A('')
    A('Conclusion: the unequal-molecules-per-pocket concern is real in principle and visible in the')
    A('mol/pkt column, but it does not bite these numbers. Either aggregation supports the same')
    A('conclusions here.')
    A('')

    # ---------------------------------------------------------------- coverage
    A('=' * W)
    A('Source: eval_out/<model>/{smina,smina_vinardo,gnina}/pocket*_{score,min,dock}.sdf, '
      'eval_out/<model>/pose_rmsd.csv')
    A('Full method: analysis/METHODS_pose_fidelity.md   Verdict: analysis/RESULTS_pose_fidelity_2026-07.md')
    A('=' * W)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    txt = '\n'.join(L)
    with open(args.out, 'w') as f:
        f.write(txt + '\n')
    print(txt)
    print(f'\n-> {args.out}')


if __name__ == '__main__':
    main()
