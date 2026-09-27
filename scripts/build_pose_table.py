#!/usr/bin/env python3
"""Pose RMSD + stability: does the generated pose survive a re-search, and is it strained?

The F1 main table (results/comparison/f1_sbdd/main_table/) answers "how tightly does it bind".
It cannot answer "is the pose real" -- a model can win every affinity column with a distorted,
clashing geometry that happens to sit at the scoring function's optimum. This is the counterweight:

    Vina | Vinardo | gnina        each as  %<2A / RMSD avg / RMSD med   (generated -> re-docked)
    PoseCheck                     clash and strain, avg and med

TWO VERSIONS, because the two halves do not lose the same molecules. PoseCheck caps the strain
relaxation at 300 s per molecule, so its coverage runs 0.1%-6.1% below the RMSD coverage:

    STRICT       a molecule needs all 3 RMSDs AND clash AND strain -> one denominator everywhere,
                 RMSD and stability pairable molecule-for-molecule.
    ENGINE-ONLY  a molecule needs all 3 RMSDs; clash/strain then use whatever subset has them,
                 each with its own n. Honest RMSD denominator, mixed denominators per row.

Two things about that split are counter-intuitive and are measured in section 0 of the output:

  * What times out is the BIGGEST molecule, not the worst one -- more atoms, more conformers to
    relax. The molecules STRICT discards are larger AND dock closer than the ones it keeps, in
    every row, so STRICT is PESSIMISTIC on RMSD, not flattering. The bias is MNAR but size-driven.
  * Every molecule with a strain value also has a clash value (0 exceptions), so STRICT is just
    3-eng INTERSECT strain and the Strain columns of the two sections are bit-identical. The whole
    difference between the versions lives in the RMSD columns.

Reads:
    results/pose_fidelity/master.csv                      per-molecule RMSD + clash + strain
    results/reference_protocol/self_redock_v4/pocket*.csv  reference RMSD  (no master.csv row)
    eval_out/native/posecheck/pocket*.csv                  reference clash/strain

Writes:
    results/comparison/f2_pose_fidelity/pose_stability/pose_stability.txt
    results/comparison/f2_pose_fidelity/pose_stability/pose_stability_strict.csv
    results/comparison/f2_pose_fidelity/pose_stability/pose_stability_engineonly.csv

Usage:
    python scripts/build_pose_table.py
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_registry                                                  # noqa: E402
from build_comparison_tables import agg                                # noqa: E402
from build_redock_comparison import read_dock_tags                     # noqa: E402

MASTER = 'results/pose_fidelity/master.csv'
REF_RMSD_DIR = 'results/reference_protocol/self_redock_v4'
REF_PC_GLOB = 'eval_out/native/posecheck/pocket*.csv'
REF_SCORE_CSV_DIR = 'results/reference_protocol/self_reference_affinity_v2'
# Clash recovered for molecules that timed out of the full PoseCheck run (scripts/eval_posecheck_clash.py).
# Applied to the ENGINE-ONLY section only -- see fill_recovered_clash.
RECOVERED_GLOB = 'eval_out/{tag}/posecheck_clash/pocket*.csv'
F1_TABLE = 'results/comparison/f1_sbdd/main_table/main_table.csv'

# Row order is FROZEN to F1's main_table so the two tables align row-for-row. It is not re-sorted
# by anything measured here -- re-sorting would mean a reader diffing the two tables could not tell
# a re-ordering from a result change.
ROW_IDS =['reference', 'pocket2mol', 'targetdiff', 'molcraft', 'pidiff_retrain',
           'ipdiff', 'alidiff', 'kgdiff', 'ours_noguide', 'ours_vina']

# The three engines, matching build_main_table.py's choice.
#
# smina-DEFAULT is deliberately absent: its scoring function IS Vina 1.1.2 (build_rmsd_summary.py:8),
# so it would duplicate the Vina column, and it has ZERO rmsd rows for ipdiff and ours_noguide --
# including it would empty the all-three intersection for both models.
#
# (record key, master.csv column, display label)
ENGINES = [('vina', 'vm_rmsd_gen_dock', 'Vina'),
           ('vinardo', 'rmsd_dock_vinardo', 'Vinardo'),
           ('gnina', 'rmsd_dock_gnina', 'gnina')]
RMSD_THRESHOLD = 2.0

# (record key, group label, [(sub-label, decimals, scientific), ...]).
#
# One group per engine spanning THREE cells -- the rate and the two moments of the same
# distribution belong under one heading, and splitting them into a 1-cell and a 2-cell group put a
# 9-character label in a 9-character slot with nowhere for a separating space.
#
# Strain's MEAN runs to 1e11-1e14: the distribution is heavy-tailed and a handful of molecules
# whose UFF relaxation diverges set it. It cannot be shown as %.3f, and dropping it would hide half
# the picture, so that one cell prints in scientific notation.
_RMSD_CELLS = [('%<2A', 1, False), ('avg', 3, False), ('med', 3, False)]
# gnina carries two extra cells: the CNN pose-quality score of the pose AS GENERATED.
# SCORE mode, not dock mode, and that is the whole point -- see load_cnnscore.
_GNINA_CELLS = _RMSD_CELLS + [('CNN av', 3, False), ('CNN md', 3, False)]
CSV_NAMES = {'%<2A': 'pct_lt2', 'CNN av': 'cnnscore_avg', 'CNN md': 'cnnscore_med'}
GROUPS = ([(e, lab, _GNINA_CELLS if e == 'gnina' else _RMSD_CELLS) for e, _, lab in ENGINES]
          + [('clash', 'Clash', [('avg', 3, False), ('med', 1, False)]),
             ('strain', 'Strain', [('avg', 2, True), ('med', 2, False)])])


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(v) else v


def cell(v, nd=3, sci=False):
    if v is None:
        return ''
    return f'{v:.2e}' if sci else f'{v:.{nd}f}'


def pct_under(vals, thr=RMSD_THRESHOLD):
    """% of measured RMSDs below thr.

    NOTE this denominator is the table's own population, in which every molecule is measured by
    every engine by construction. build_rmsd_summary.frac_under() uses a WIDER denominator -- every
    docking-successful molecule, counting an unmeasured one as a failure -- so the two differ
    slightly by design. Coverage is >=99.9% everywhere, so they should agree to <0.1 pp.
    """
    return 100.0 * float(np.mean(np.array(vals) < thr)) if vals else None


def load_models():
    """-> {eval_out tag: [record, ...]} over docking-successful molecules only."""
    by_tag = {}
    for r in csv.DictReader(open(MASTER)):
        if r.get('orig_docked') != '1':
            continue
        rec = {'name': r.get('name'), 'recovered': False,
               'pk': r.get('pocket_idx'), 'heavy': _f(r.get('heavy')),
               'gnina_method': r.get('method_dock_gnina') or '',
               'clash': _f(r.get('pc_clashes')), 'strain': _f(r.get('pc_strain')),
               'cnnscore': None}
        for key, col, _ in ENGINES:
            rec[key] = _f(r.get(col))
        by_tag.setdefault(r['model'], []).append(rec)
    return by_tag


def load_cnnscore(tag):
    """{molecule: CNNscore} for the pose AS GENERATED (gnina --score_only).

    SCORE mode, deliberately, not dock. gnina picks its single dock pose BY CNNscore, so the
    dock-mode value is near-maximal by construction and says more about the selection rule than
    about the molecule -- measured, it compresses the models into 0.675-0.759 while the score-mode
    value spreads them over 0.508-0.712. This table grades the pose the MODEL produced, so it wants
    the score-mode number, which is also the only one of the two that is not circular.

    Coverage is 100% of the 3-engine population in every model, so this adds no population change.
    """
    raw = read_dock_tags(os.path.join('eval_out', tag), 'gnina', mode='score')
    return {k: _f(v.get('CNNscore')) for k, v in raw.items() if _f(v.get('CNNscore')) is not None}


def load_reference():
    """The reference row. `native` has NO row in master.csv, so this is a second code path.

    RMSD comes from the self-redock protocol (the crystal ligand put through the IDENTICAL commands
    as the model rows) and clash/strain from eval_out/native/posecheck/. Both are keyed by the
    pocket index in the filename, one molecule per pocket.

    Only self_redock_v4 may be read: the un-suffixed self_redock/ is the pre-fix instrumentation in
    which 66/100 pockets fell back to skeleton isomorphism (build_rmsd_summary.py:23-35).
    """
    recs = {}
    for f in sorted(glob.glob(f'{REF_RMSD_DIR}/pocket*.csv')):
        for r in csv.DictReader(open(f)):
            pk = int(r['pocket_idx'])
            recs[pk] = {'pk': str(pk), 'heavy': _f(r.get('heavy')), 'gnina_method': '',
                        'vina': _f(r.get('rmsd_vina_meeko')),
                        'vinardo': _f(r.get('rmsd_vinardo')),
                        'gnina': _f(r.get('rmsd_gnina')),
                        'clash': None, 'strain': None, 'cnnscore': None}
    for f in sorted(glob.glob(f'{REF_SCORE_CSV_DIR}/pocket*.csv')):
        for r in csv.DictReader(open(f)):
            pk = int(r['pocket_idx'])
            if pk in recs:
                recs[pk]['cnnscore'] = _f(r.get('gnina_CNNscore'))
    for f in sorted(glob.glob(REF_PC_GLOB)):
        pk = int(os.path.basename(f)[len('pocket'):].split('.')[0])
        for r in csv.DictReader(open(f)):
            if pk in recs:
                recs[pk]['clash'] = _f(r.get('clashes'))
                recs[pk]['strain'] = _f(r.get('strain_energy'))
    return list(recs.values())


def fill_recovered_clash(recs, tag):
    """Fill clash for molecules whose PoseCheck run was killed by the strain timeout.

    Those molecules lost clash only as collateral: clash is a distance count that never failed,
    but the child process that would have written it was killed while the UFF strain relaxation
    ran over. They are the LARGEST molecules (29-36 heavy atoms against 24 for the survivors), and
    clash grows with size, so leaving them out biases every clash mean downward by a
    model-dependent amount. scripts/eval_posecheck_clash.py recomputes just the clash for them.

    STRAIN IS NOT FILLED, so this cannot leak into the STRICT section: STRICT requires clash AND
    strain, and these molecules still have no strain. Their clash therefore reaches the ENGINE-ONLY
    section -- which aggregates each stability column over its own subset -- and nothing else.
    """
    rec_map = {}
    for f in sorted(glob.glob(RECOVERED_GLOB.format(tag=tag))):
        for r in csv.DictReader(open(f)):
            v = _f(r.get('clashes'))
            if v is not None:
                rec_map[r['molecule']] = v
    n = 0
    for r in recs:
        if r['clash'] is None and r.get('name') in rec_map:
            r['clash'] = rec_map[r['name']]
            r['recovered'] = True
            n += 1
    return n


def populations(recs):
    """-> (engine_only, strict). Both are lists of records."""
    eng = [r for r in recs if all(r[k] is not None for k, _, _ in ENGINES)]
    strict = [r for r in eng if r['clash'] is not None and r['strain'] is not None]
    return eng, strict


def delta_lines(strict_rows, eng_rows):
    """STRICT - ENGINE per cell, blank where zero, plus any rank flip the choice causes.

    Exists because the two sections are 14 columns wide and near-identical: without this the reader
    has to eyeball 140 numbers across a page break to discover that the Strain columns never move
    and the RMSD columns move by at most ~1 pp.
    """
    L = ['A blank cell means the two sections agree exactly.', '']
    hdr = f'{"Model":<{LW}}'
    for _, lab, cells in GROUPS:
        # composite label must leave a separating space, so cap it at CW-1
        hdr += ''.join(f'{(lab.split()[0][:3] + "." + s)[:CW - 1]:>{CW}}' for s, _, _ in cells)
    L += [hdr.rstrip(), '-' * W]
    si = {r[0]: r for r in strict_rows}
    flips = []
    for mid, label, erow, _, _, _ in eng_rows:
        srow = si[mid][2]
        line = f'{label:<{LW}}'
        for key, _, cells in GROUPS:
            for i, (_, nd, _) in enumerate(cells):
                a, b = srow[key][i], erow[key][i]
                d = None if a is None or b is None else a - b
                line += f'{"" if d is None or abs(d) < 10 ** -nd / 2 else f"{d:+.{nd}f}":>{CW}}'
        L.append(line.rstrip())

    # A rank flip is the only way this choice can change a sentence in the paper.
    for key, _, cells in GROUPS:
        for i, (sub, nd, _) in enumerate(cells):
            def order(rows):
                v = [(r[1], r[2][key][i]) for r in rows if r[2][key][i] is not None]
                return [n for n, _ in sorted(v, key=lambda x: -x[1])]
            oe, os_ = order(eng_rows), order(strict_rows)
            for a in range(len(oe)):
                for b in range(a + 1, len(oe)):
                    x, y = oe[a], oe[b]
                    if os_.index(x) > os_.index(y):
                        flips.append(f'{key} {sub}: {x} vs {y} -- ENGINE says {x}, STRICT says {y}')
    L += ['']
    if flips:
        L += ['RANK FLIPS CAUSED BY THE DENOMINATOR CHOICE:'] + [f'  {f}' for f in sorted(set(flips))]
    else:
        L += ['No pair of models changes order between the two sections.']
    L += ['',
          'PUT THESE NEXT TO THE PUBLISHED CONFIDENCE INTERVAL BEFORE READING ANYTHING INTO THEM.',
          'f2_pose_fidelity/rmsd_summary.txt section 3a gives a 95% pocket-level bootstrap of about',
          '+/-5 pp on these rates -- e.g. KGDiff Vinardo 56.2% [51.0,61.2] and Ours (Vina Only)',
          '56.0% [50.7,61.4]. Every delta in this section is an order of magnitude smaller than that',
          'interval, so NO claim in this file may rest on the choice between its two sections.', '']
    return L


def build_row(recs):
    """-> (row, n_pk, n_mol, diag)."""
    row = {}
    for key, _, _ in ENGINES:
        vals = [r[key] for r in recs if r[key] is not None]
        row[key] = (pct_under(vals),) + agg(vals)
        if key == 'gnina':
            row[key] += agg([r['cnnscore'] for r in recs if r.get('cnnscore') is not None])
    for m in ('clash', 'strain'):
        vals = [r[m] for r in recs if r[m] is not None]
        row[m] = agg(vals)
    heavy = [r['heavy'] for r in recs if r['heavy'] is not None]
    gm = [r['gnina_method'] for r in recs if r['gnina_method']]
    diag = {
        'n_clash': sum(1 for r in recs if r['clash'] is not None),
        'n_clash_rec': sum(1 for r in recs if r['clash'] is not None and r.get('recovered')),
        'n_strain': sum(1 for r in recs if r['strain'] is not None),
        'heavy_mean': float(np.mean(heavy)) if heavy else None,
        # skeleton_iso is the bond-order-agnostic fallback; it can only bias RMSD DOWNWARD, and it
        # fires only for baselines, so a model with more of it gets a small unearned advantage.
        'iso_pct': 100.0 * sum(1 for m in gm if m == 'skeleton_iso') / len(gm) if gm else None,
    }
    return row, len({r['pk'] for r in recs}), len(recs), diag


# ----------------------------------------------------------------------------- output

LW = 22
CW = 9
W = LW + 6 + 7 + CW * sum(len(cells) for _, _, cells in GROUPS)


def render(rows):
    """Two-level header; the group label is centred over its whole span."""
    head = f'{"Model":<{LW}}{"n_pk":>6}{"n_mol":>7}'
    sub = f'{"":<{LW}}{"":>6}{"":>7}'
    for _, lab, cells in GROUPS:
        head += f'{lab:^{len(cells) * CW}}'
        sub += ''.join(f'{s:>{CW}}' for s, _, _ in cells)
    lines = [head.rstrip(), sub, '-' * W]
    for label, row, n_pk, n_mol in rows:
        line = f'{label:<{LW}}{n_pk:>6}{n_mol:>7}'
        for key, _, cells in GROUPS:
            for v, (_, nd, sci) in zip(row[key], cells):
                line += f'{cell(v, nd, sci):>{CW}}'
        lines.append(line.rstrip())
    return lines


def csv_header():
    cols = ['rank', 'id', 'label', 'n_pockets', 'n_mols']
    for key, _, cells in GROUPS:
        cols += [f'{key}_{CSV_NAMES.get(s, s)}' for s, _, _ in cells]
    return cols + ['n_clash', 'n_strain', 'heavy_mean', 'gnina_skeleton_iso_pct',
                   'n_3eng', 'n_strict']


def write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(csv_header())
        for rank, (mid, label, row, n_pk, n_mol, d) in enumerate(rows, 1):
            vals = []
            for key, _, cells in GROUPS:
                # CSV keeps plain decimals; the scientific formatting is a display concern only.
                vals += [('' if v is None else f'{v:.6g}') if sci else cell(v, nd)
                         for v, (_, nd, sci) in zip(row[key], cells)]
            w.writerow([rank, mid, label, n_pk, n_mol] + vals
                       + [d['n_clash'], d['n_strain'], cell(d['heavy_mean'], 1),
                          cell(d['iso_pct'], 2), d['n_3eng'], d['n_strict']])


def f1_order():
    """Row order from F1's main_table, so the two tables line up. Falls back to ROW_IDS."""
    if not os.path.isfile(F1_TABLE):
        print(f'[warn] {F1_TABLE} missing -- using the frozen ROW_IDS order')
        return ROW_IDS
    ids = [r['id'] for r in csv.DictReader(open(F1_TABLE))]
    if [i for i in ids if i in ROW_IDS] != [i for i in ROW_IDS if i in ids]:
        print('[warn] F1 main_table row order differs from ROW_IDS; using F1 order')
    return [i for i in ids if i in ROW_IDS] + [i for i in ROW_IDS if i not in ids]


def header_lines(cov):
    bar, sub = '=' * W, '-' * W
    L = [bar, 'POSE RMSD + STABILITY', bar, '',
         'RMSD is Angstrom, LOWER better. %<2A HIGHER better. Clash and Strain LOWER better.', '',
         'RMSD    non-superposed, symmetry-minimised heavy-atom RMSD (rdMolAlign.CalcRMS) from the',
         '        molecule AS GENERATED to that engine\'s own RE-DOCKED pose. It is NOT a distance',
         '        to the crystal pose. Superposing (GetBestRMS) is deliberately not used: it would',
         '        turn "how far did the pose move in the pocket" into "how different is the shape".',
         '%<2A    share of the row\'s molecules with that RMSD below 2 A. Single pooled rate.',
         'CNN av/md  gnina CNNscore of the pose AS GENERATED (--score_only): the CNN\'s',
         '        probability in [0,1] that the pose is right, so HIGHER is better -- the opposite',
         '        direction to the RMSD cells sitting beside it in the same group. It is the SCORE',
         '        mode on purpose: gnina selects its single dock pose BY CNNscore, so the dock-mode',
         '        value is near-maximal by construction and compresses the models into 0.675-0.759,',
         '        while the score mode spreads them over 0.508-0.712 and is not circular. This is',
         '        the only column here that grades the generated pose without re-docking it first.',
         'Clash   count of protein-ligand ATOM PAIRS closer than Rvdw(i)+Rvdw(j)-0.5 A. A PAIR',
         '        count, NOT normalised by ligand or pocket size.',
         'Strain  E(pose, locally relaxed) - min E(51 freely relaxed conformers), UFF, kcal/mol.',
         '', sub,
         'HOW TO READ THIS TABLE WITHOUT GETTING IT BACKWARDS', sub, '',
         '1. A LOW dock-RMSD IS NOT "MATCHES THE CRYSTAL POSE". Every model here re-docks CLOSER to',
         '   its own generated pose than the reference ligand re-docks to its own. That is not the',
         '   models beating a crystal structure -- it means their poses already sit at the scoring',
         '   function\'s global optimum, which the crystal ligand does not. Quote a low dock-RMSD as',
         '   "at the docking optimum", never as "matches the crystal pose".',
         '',
         '2. THE RMSD COLUMNS ARE PARTLY CIRCULAR FOR OUR MODELS. They are trained on a Vina-derived',
         '   physics loss, and all three engines search with a Vina-family function (gnina\'s default',
         '   --cnn_scoring rescore uses the CNN only to re-rank the final pose). A model optimised',
         '   against the scorer will land near the scorer\'s optimum; that is close to tautological.',
         '',
         '3. STRAIN: READ THE MEDIAN. The distribution is heavy-tailed -- a handful of molecules with',
         '   a diverging UFF relaxation drive the mean to 1e11-1e14. The mean is printed (in',
         '   scientific notation) so the tail is not hidden, but the median is the quotable number.',
         '',
         '4. SIZE CONFOUNDS BOTH HALVES. RMSD and strain both grow with molecule size, so a model',
         '   that makes small rigid molecules wins without being better. Pocket2Mol averages 17.7',
         '   heavy atoms against 24+ for everyone else -- heavy_mean is in section 0 for this reason.',
         '',
         '5. gnina\'s skeleton_iso FALLBACK IS PRESENT BUT NEGLIGIBLE -- do not use it to discount',
         '   the column. When an engine re-perceived bond orders, RMSD falls back to a',
         '   bond-order-agnostic skeleton isomorphism, which in principle can only pull a value DOWN,',
         '   and it fires only in the gnina arm and only for baselines (TargetDiff 3.08%, KGDiff',
         '   1.61%, both of ours 0%). Measured, dropping every such row moves gnina %<2A by',
         '   +0.02 to +0.06 pp -- TargetDiff 33.62 -> 33.66, KGDiff 43.74 -> 43.80. And those rows\'',
         '   own median RMSD is HIGHER, not lower (TargetDiff 3.73 vs 3.45, AliDiff 5.54 vs 4.03):',
         '   the fallback fires on harder molecules, so the theoretical downward bias does not',
         '   dominate what it actually measures. Per-model rate in section 0.',
         '',
         '6. THE ENGINES USE DIFFERENT BOXES, so "dock" is not quite the same search in each column:',
         '   Vinardo/gnina --autobox_ligand <crystal ligand> at exhaustiveness 8, AutoDock Vina',
         '   (meeko) a fixed 20 A cube at exhaustiveness 16. RMSD is in Angstrom, so unlike the',
         '   affinity columns of the F1 table these ARE comparable across engines -- but a column',
         '   is still measuring its own engine\'s idea of where the molecule belongs.',
         '',
         '7. ESTIMATOR: molecule-pooled mean/median throughout, NOT per-pocket. There are no',
         '   confidence intervals here; f2_pose_fidelity/rmsd_summary.txt carries the pocket-level',
         '   bootstrap CIs and is the file to cite for any significance claim.',
         '',
         '8. THE REFERENCE ROW IS NOT A CEILING HERE, despite being labelled one elsewhere. It is',
         '   1 molecule per pocket (n=100), so its per-molecule and per-pocket statistics coincide,',
         '   and it is a DOCKED/MINIMISED CrossDocked pose (64/100 cross-docked), not a crystal',
         '   structure. It is beaten on 3 of the 5 metric groups: gnina %<2A 52.0 vs our 55.0 and',
         '   53.6; clash 7.92 vs PIDiff 6.79 and both our arms; strain median 9.16 vs Pocket2Mol',
         '   4.03. That is point 1 restated -- a real binding mode is not where a docking scorer',
         '   puts its optimum, so re-searching moves it further than it moves a generated pose.',
         '', sub,
         'DO NOT PUT THESE NUMBERS IN A TABLE WITH ANOTHER PAPER\'S', sub, '',
         'Clash, Strain and RMSD<2A are all named the same way across the SBDD literature and are',
         'all measured differently. MolJO (Table 14) reports the SAME 100 CrossDocked reference',
         'ligands through the SAME tool (PoseCheck, Harris et al. 2023) and does not agree with us:',
         '',
         '                        MolJO Table 14      this table',
         '    Reference strain               114            9.16      (12x)',
         '    Reference clash               5.46            7.92      (1.45x)',
         '    Reference RMSD<2A            34.0%           26.0%',
         '    TargetDiff strain             1208          179.47      (6.7x)',
         '    MolCRAFT   strain              196           18.48      (10.6x)',
         '    IPDiff     strain             5861          366.85      (16x)',
         '    Pocket2Mol strain              186            4.03      (46x)',
         '',
         'The ratio is not constant, so this is not a unit conversion -- the two pipelines differ in',
         'setup. The RANKING largely survives (IPDiff worst, TargetDiff next, in both), the ABSOLUTE',
         'VALUES do not, and even the ranking breaks at the top: MolJO has Reference better than',
         'Pocket2Mol on strain (114 < 186) while we have the reverse (9.16 > 4.03).',
         '',
         'RMSD<2A is worse still, because the DEFINITION differs. MolJO states (Appendix G): "not',
         'all pose pairs are available for calculating symmetry-corrected RMSD, where we report the',
         'non-corrected RMSD instead to make sure that all samples are faithfully evaluated." This',
         'table uses symmetry-minimised RMSD throughout and treats an unmeasurable pair as',
         'not-measured. Symmetry correction is not a rounding difference: on one molecule of ours the',
         'index-order value is 6.69 A against 0.064 A corrected. Their choice keeps every sample at',
         'the cost of inflating RMSD; ours keeps the definition at the cost of the denominator.',
         '',
         'Quote another paper\'s clash / strain / RMSD<2A only as that paper\'s own internal ranking.',
         'To place a competitor in THIS table, its generated molecules must be re-measured through',
         'this pipeline -- which is what was done for the eight baselines here.',
         '', sub, '0. COVERAGE -- WHY THERE ARE TWO TABLES', sub, '',
         'docked   docking-successful molecules (the repo-wide population rule)',
         '3-eng    of those, measured by ALL THREE engines      -> section 2 (ENGINE-ONLY)',
         'clash    of the 3-eng set, how many have a clash value',
         'strain   of the 3-eng set, how many have a strain value',
         'STRICT   of the 3-eng set, how many have BOTH          -> section 1 (STRICT)',
         'lost%    (3-eng - STRICT) / 3-eng, i.e. what STRICT discards',
         'iso%     share of the gnina arm using the skeleton_iso fallback (biases RMSD down)',
         '']
    L.append(f'{"Model":<{LW}}{"docked":>8}{"3-eng":>8}{"clash":>8}{"strain":>8}{"STRICT":>8}'
             f'{"lost%":>7}{"iso%":>7}{"heavy":>7}{"n_pk":>6}')
    L.append('-' * W)
    for c in cov:
        lost = 100.0 * (c['n_3eng'] - c['n_strict']) / c['n_3eng'] if c['n_3eng'] else 0.0
        L.append(f'{c["label"]:<{LW}}{c["n_docked"]:>8}{c["n_3eng"]:>8}{c["n_clash"]:>8}'
                 f'{c["n_strain"]:>8}{c["n_strict"]:>8}{lost:>7.1f}'
                 f'{cell(c["iso_pct"], 2):>7}{cell(c["heavy_mean"], 1):>7}{c["n_pk"]:>6}')
    L += ['', 'CLASH RECOVERY (applies to section 2 only).', '',
          'A timed-out molecule lost its clash value only as collateral -- clash is a distance count',
          'that never failed, but the child process that would have written it was killed while the',
          'UFF strain relaxation ran over. scripts/eval_posecheck_clash.py recomputes just the clash',
          'for those molecules, off the same patched prolif stack, writing to a separate directory so',
          'the pinned posecheck/ measurement is untouched. 1616 of 1616 recovered, 0 failures.',
          '',
          'It is applied ONLY to the ENGINE-ONLY section: STRICT requires clash AND strain, and these',
          'molecules still have no strain, so section 1 is unchanged by construction. Strain is not',
          'recovered -- see the bound below.',
          '']
    L.append(f'{"Model":<{LW}}{"n_rec":>8}{"clash n":>9}{"was":>9}{"gap":>6}{"clash avg":>11}'
             f'{"was":>9}{"shift":>9}')
    L.append('-' * W)
    for c in cov:
        pre, npre = c.get('clash_pre', (None, 0))
        cur = c.get('clash_now')
        sh = (cur - pre) if (cur is not None and pre is not None) else None
        gap = c['n_3eng'] - c['n_clash']
        L.append(f'{c["label"]:<{LW}}{c.get("n_rec", 0):>8}{c["n_clash"]:>9}{npre:>9}{gap:>6}'
                 f'{cell(cur, 3):>11}{cell(pre, 3):>9}{("" if sh is None else f"{sh:+.3f}"):>9}')
    L += ['',
          'WHAT THE RECOVERED MOLECULES LOOK LIKE. They are much bigger than the ones that never',
          'timed out -- that is WHY they timed out -- and in 8 of the 9 models they also clash more:',
          '',
          '    model                    kept clash  kept heavy    recovered clash  recovered heavy',
          '    pocket2mol                    8.258        17.7             11.207             31.4',
          '    targetdiff                   13.688        24.1             19.988             35.0',
          '    molcraft                      9.549        22.7             10.100             34.5',
          '    pidiff                        6.791        24.4             12.316             35.7',
          '    ipdiff                       12.542        23.8             16.756             33.1',
          '    alidiff                      11.397        24.1             13.960             31.9',
          '    kgdiff                       10.253        24.0             13.237             32.6',
          '    ours_noguide                  7.792        23.6              9.059             31.1',
          '    ours_vina                     7.473        24.2              7.054             30.8   <-- LOWER',
          '',
          'Ours (Vina+Guide) is the single exception, and it is not a rounding artefact: its',
          'recovered molecules average 30.8 heavy atoms against 24.2 for the rest of its own set --',
          'a 27% size increase -- yet clash slightly LESS (7.054 vs 7.473). Every other model gets',
          'markedly worse over the same size jump; TargetDiff goes 13.7 -> 20.0. So the correction',
          'moves 8 models up and this one very slightly down (-0.010), and our clash margin widens',
          'rather than shrinks. That was NOT the expected direction -- the prior was that our arms',
          'carry more timeouts and would therefore give up more.',
          '',
          'gap = molecules in the 3-engine set that STILL have no clash after recovery. Recovery',
          'closes the TIMEOUT gap, not every gap: a handful of molecules produced no PoseCheck row',
          'for other reasons (receptor/ligand parse failures), and those are not timeouts and are',
          'not recoverable this way. They are left as not-measured rather than imputed.',
          '',
          '',
          'THE LOSS IS NOT RANDOM, AND IT DOES NOT POINT THE WAY YOU WOULD EXPECT. PoseCheck caps',
          'the strain relaxation at 300 s per molecule. What stalls is not the worst geometry, it is',
          'the BIGGEST molecule -- more atoms and more conformers to relax. Measured, the molecules',
          'STRICT discards are LARGER and dock BETTER than the ones it keeps, in every single row:',
          '',
          '    model                 kept n  drop n   kept Vinardo med   DROPPED Vinardo med   kept/drop heavy',
          '    pocket2mol              9801      30              3.748                 1.874      17.7 / 30.5',
          '    targetdiff              8954      82              3.615                 1.906      24.1 / 35.0',
          '    molcraft                9657      10              3.188                 1.960      22.7 / 34.5',
          '    pidiff                   831      19              3.293                 1.507      24.4 / 35.7',
          '    ipdiff                  8689     317              3.410                 0.864      23.8 / 32.6',
          '    alidiff                 8858     374              2.346                 0.829      24.1 / 31.8',
          '    kgdiff                  8278     535              1.393                 0.507      24.0 / 32.5',
          '    vina_fixed_best_noguide 2111      34              1.422                 0.622      23.6 / 31.1',
          '    vina_fixed              8071     266              0.749                 0.456      24.2 / 29.0',
          '',
          'So STRICT is not a flattering subset -- it is a PESSIMISTIC one on the RMSD half. Its',
          '%<2A is lower than section 2\'s in 9 of 10 rows and its medians are higher, because the',
          'molecules it drops are the ones that re-docked closest. The bias is real and it is MNAR,',
          'but it is driven by SIZE, not by badness.',
          '',
          'AND THE STABILITY HALF DOES NOT MOVE AT ALL. Every molecule with a strain value also has',
          'a clash value (verified: 0 exceptions across all 10 tags), so STRICT = 3-eng INTERSECT',
          'strain, and the Strain columns of sections 1 and 2 are BIT-IDENTICAL. Clash differs by at',
          'most 0.004. The entire difference between the two sections lives in the RMSD columns --',
          'the opposite of what the two-version split was set up to examine.',
          '',
          'WHICH ONE TO QUOTE. Use section 2 (ENGINE-ONLY) for any RMSD claim: it reproduces',
          'f2_pose_fidelity/rmsd_summary.csv to within 0.02 pp. Use section 1 (STRICT) only when you',
          'need to pair an RMSD with a clash or strain molecule-for-molecule. No claim should depend',
          'on the choice -- see section 3.',
          '',
          'IS IT WORTH RE-MEASURING THE TIMED-OUT MOLECULES? For STRAIN, provably not. Placing every',
          'timed-out molecule at +infinity -- the worst case anything could recover -- moves the',
          'medians but leaves the model ORDER completely unchanged:',
          '',
          '    model              t/o   strain med   worst case    shift        clash avg   worst case',
          '    Pocket2Mol          29          4.0          4.1     +0.1             8.26         8.49',
          '    MolCRAFT            10         18.5         18.5     +0.1             9.55         9.70',
          '    TargetDiff          82        179.5        185.5     +6.0            13.69        16.30',
          '    IPDiff             307        366.8        446.5    +79.7            12.54        17.97',
          '    PIDiff              19        544.3        593.3    +49.0             6.79         9.28',
          '    Ours (Vina+Guide)  205        565.7        611.1    +45.4             7.47        12.74',
          '    Ours (Vina Only)    34        624.1        658.4    +34.4             7.79        10.08',
          '    KGDiff             531        643.5        867.7   +224.2            10.25        23.07',
          '    AliDiff            372       1832.0       2148.5   +316.6            11.40        21.33',
          '',
          '    measured   order: Pocket2Mol < MolCRAFT < TargetDiff < IPDiff < PIDiff < Ours(V+G)',
          '                      < Ours(V) < KGDiff < AliDiff',
          '    worst-case order: IDENTICAL',
          '',
          'So no strain conclusion in this file can be overturned by recovering those molecules, and',
          'a longer timeout would not be a cheap run -- the UFF relaxation on these can diverge, which',
          'is why they timed out. CLASH WAS A DIFFERENT MATTER and HAS BEEN FIXED: it is purely',
          'geometric (a distance count, no force field), so it was never the expensive part -- it',
          'was missing only because the timeout kills the whole per-molecule PoseCheck run and no',
          'row is written. Its worst-case bound above was NOT tight (Ours 7.47 -> 12.74 crossed',
          'Pocket2Mol 8.49), so it was recomputed rather than bounded: all 1616 timed-out molecules',
          'were re-run with the strain step disabled, 0 failures. The clash column of section 2 is',
          'that corrected number; see CLASH RECOVERY below. Strain remains bounded rather than',
          're-measured, because re-running it is expensive and provably cannot change a ranking.', '']
    return L


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out_dir', default='results/comparison/f2_pose_fidelity/pose_stability')
    p.add_argument('--registry', default='configs/models.json')
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    by_tag = load_models()
    order = f1_order()
    built = []
    for mid in order:
        e = model_registry.by_id(mid, a.registry)
        label = model_registry.label_for(e, 'sbdd')
        tag = (e.get('ids') or {}).get('eval_out')
        recs = load_reference() if e['tier'] == 'reference' else by_tag.get(tag, [])
        if not recs:
            print(f'[warn] {label}: no rows (tag={tag}) -- skipped')
            continue
        # Baseline must be measured on the SAME population the recovered figure will be reported
        # on (the 3-engine set), not on all docking-successful molecules -- otherwise the "was"
        # column and the "now" column have different denominators and the shift is not a shift.
        if e['tier'] != 'reference':
            cnn = load_cnnscore(tag)
            for r in recs:
                r['cnnscore'] = cnn.get(r.get('name'))
        eng_pre, _ = populations(recs)
        pre = [r['clash'] for r in eng_pre if r['clash'] is not None]
        n_rec = 0 if e['tier'] == 'reference' else fill_recovered_clash(eng_pre, tag)
        eng, strict = populations(recs)
        built.append({'id': mid, 'label': label, 'n_docked': len(recs), 'n_rec': n_rec,
                      'clash_pre': (float(np.mean(pre)) if pre else None, len(pre)),
                      'eng': eng, 'strict': strict})
        print(f'  {label:22s} docked={len(recs):5d}  3-eng={len(eng):5d}  strict={len(strict):5d}')

    cov, sec = [], {}
    for kind in ('strict', 'eng'):
        sec[kind] = []
        for b in built:
            recs = b[kind]
            row, n_pk, n_mol, d = build_row(recs)
            d['n_3eng'], d['n_strict'] = len(b['eng']), len(b['strict'])
            sec[kind].append((b['id'], b['label'], row, n_pk, n_mol, d))
            if kind == 'eng':
                cov.append({'label': b['label'], 'n_docked': b['n_docked'], 'n_pk': n_pk,
                            'n_rec': b['n_rec'], 'clash_pre': b['clash_pre'],
                            'clash_now': row['clash'][0],
                            'n_3eng': len(b['eng']), 'n_strict': len(b['strict']),
                            'n_clash': d['n_clash'], 'n_strain': d['n_strain'],
                            'heavy_mean': d['heavy_mean'], 'iso_pct': d['iso_pct']})

    L = header_lines(cov)
    L += ['-' * W, '1. STRICT -- every column on ONE denominator', '-' * W, '',
          'A molecule counts only if all three RMSDs AND clash AND strain exist. RMSD and strain',
          'can therefore be paired molecule-for-molecule. See section 0 for what this discards.', '']
    L += render([(lb, r, pk, nm) for _, lb, r, pk, nm, _ in sec['strict']])
    L += ['', '-' * W, '2. ENGINE-ONLY -- RMSD denominator kept whole', '-' * W, '',
          'A molecule counts if all three RMSDs exist. Clash and Strain then aggregate over the',
          'subset of those molecules that has them, so THOSE TWO COLUMNS HAVE A SMALLER n THAN THE',
          'ROW\'S n_mol -- the per-column counts are in section 0 (clash / strain).', '']
    L += render([(lb, r, pk, nm) for _, lb, r, pk, nm, _ in sec['eng']])
    L += ['', '-' * W, '3. WHAT THE DENOMINATOR CHOICE ACTUALLY CHANGES  (section 1 - section 2)',
          '-' * W, '']
    L += delta_lines(sec['strict'], sec['eng'])
    L += ['',
          'Source:     scripts/build_pose_table.py',
          f'            {MASTER}; reference from {REF_RMSD_DIR} + {REF_PC_GLOB}',
          'Regenerate: python scripts/build_pose_table.py', '']

    txt = os.path.join(a.out_dir, 'pose_stability.txt')
    with open(txt, 'w') as f:
        f.write('\n'.join(L) + '\n')
    cs = os.path.join(a.out_dir, 'pose_stability_strict.csv')
    ce = os.path.join(a.out_dir, 'pose_stability_engineonly.csv')
    write_csv(cs, sec['strict'])
    write_csv(ce, sec['eng'])
    print(f'\nWrote:\n  {txt}\n  {cs}\n  {ce}')


if __name__ == '__main__':
    main()
