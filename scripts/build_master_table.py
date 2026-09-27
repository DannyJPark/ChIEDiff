#!/usr/bin/env python3
"""One row per model, every family's headline metrics, joined on the registry `id`.

    python scripts/build_master_table.py
    python scripts/build_master_table.py --all_columns     # every column, not just the headline

Outputs `results/comparison/master_table.{csv,txt}`.

WHY THIS COULD NOT EXIST BEFORE. The same model was known by up to six strings -- our main model
was `vina_fixed` / `vina_fixed_best` / `vina_fixed_on` / `Ours (vina_fixed_best)` /
`Ours (guidance ON)` / `Ours (Vina loss)`. Joining five tables on a display name produced six
garbage rows. Every family CSV now carries the registry `id`, and that is the ONLY key used here.

WHAT A BLANK MEANS. Blank is never "zero" and never "bad". The registry declares why, per family:

    ok              measured and in this family's main table
    ok[x-abl]       measured, but rendered in sub-table x, not the main csv
                    (a blank beside a bare `ok` IS a bug; beside ok[..] it is not)
    not_measured    no data on disk, and re-measuring was out of scope
    not_applicable  structurally impossible (PIDiff has 8.5 mols/pocket vs the 20 F4 needs)
    deferred        data exists, wiring the row was postponed
    retired         measured and kept, deliberately withdrawn from reporting
    (blank)         the model is not part of that family at all

The status column sits next to the metrics so a reader can never mistake an empty cell for a
measured zero. The row axis is the REGISTRY, not any one table, so a model measured nowhere still
appears -- an absence you can see beats an absence you have to notice.

CROSS-COLUMN WARNINGS, carried into the text output because this is exactly where a reader is
tempted to ignore them:
  * F2 RMSD columns use different boxes and search depths per engine. Rank within a column.
  * F3 PLIP counts atom pairs; ProLIF caps at one per (residue, type). Never sum or difference
    them. ProLIF is version-locked to prolif 2.2.0 and is not comparable to other papers.
  * F4 Delta is ~-0.87 correlated with the absolute on-target score, so read `off_med` beside it.
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_registry                                                    # noqa: E402

ROOT = 'results/comparison'

# (family, csv path, row filter, headline columns, prefix, estimator, population)
#
# `estimator` and `population` are NOT decoration. The families do not compute the same kind of
# average over the same molecules, and the table has to say so where the numbers are, or it
# invites exactly the cross-family comparison that is invalid:
#
#   f1 vina_dock_avg   mean over MOLECULES pooled across pockets
#   f4 delta_p_avg     mean of PER-POCKET means (SBE-Diff Eq.6, 1/m_i * sum_j)
#
# Those are different estimators. Ranking a model by one against another model by the other is
# meaningless. Every family restricts to docking-successful molecules -- the RULE is shared --
# but the realised n is not, because each instrument loses its own molecules on top.
SOURCES = [
    ('sbdd', f'{ROOT}/f1_sbdd/comparison_overall.csv', None,
     ['n_pockets', 'n_mols', 'vina_dock_avg', 'vina_dock_med', 'qed_med', 'sa_med', 'div_med',
      'ha_pct_med'], 'f1',
     'molecule-pooled (avg/med over all molecules of all pockets)',
     'docking-successful only'),

    ('pose', f'{ROOT}/f2_pose_fidelity/rmsd_summary.csv', None,
     ['rmsd_dock_vina_n', 'rmsd_dock_vinardo_n', 'rmsd_dock_gnina_n',
      'rmsd_dock_vina_med', 'rmsd_dock_vinardo_med', 'rmsd_dock_gnina_med',
      'rmsd_min_vinardo_med', 'rmsd_dock_vinardo_pct_lt2'], 'f2',
     'pooled-molecule MEDIAN, 95% CI resampled over POCKETS (point estimate and interval '
     'are on different units)',
     'docking-successful only; n differs per ENGINE -- all three n columns are shown for that '
     'reason, they are not redundant'),

    ('nci', f'{ROOT}/f3_nci/nci_summary.csv', None,
     ['n_docked', 'n_plip', 'n_prolif', 'heavy_mean',
      'plip_n_hbond', 'plip_n_hydrophobic', 'plip_total', 'plip_per_heavy',
      'prolif_HBAcceptor', 'prolif_HBDonor', 'prolif_Hydrophobic', 'prolif_total',
      'prolif_per_heavy'], 'f3',
     'molecule-pooled MEAN. nci_summary.txt section 2b carries the pocket-level aggregation '
     'and its cluster-bootstrap CI -- use THAT for any model-vs-model claim',
     'docking-successful only, but PLIP and ProLIF have DIFFERENT n: PoseCheck/ProLIF loses '
     'molecules to strain-relaxation timeouts that PLIP does not'),

    # model x mode x arm. The slice taken is Vina Dock on the ROSTER arm: the pockets the
    # nine REPORTED models share (99 -- only pocket 46, empty for both PIDiff and ours, is
    # dropped). The `common` arm also intersects AR and DecompDiff, which appear in no
    # manuscript table, and their gaps at pockets 25/58/82/83 cost every reported row four
    # pockets. Still deliberately NOT 100: widening further would compare rows measured on
    # different pocket sets.
    ('delta', f'{ROOT}/f4_delta_score/delta_score_overall.csv',
     lambda r: r.get('mode') == 'dock' and r.get('arm') == 'roster',
     ['n_pk', 'n_mol', 'delta_p_avg', 'delta_p_med', 'on_med', 'off_med',
      'ratio_pct_avg', 'ratio_pct_med', 'win_pct_avg', 'win_pct_med'], 'f4',
     'mean/median of PER-POCKET means (SBE-Diff Eq.6) -- NOT molecule-pooled like f1',
     'docking-successful AND has a 3-D mol, restricted to the 99 pockets every REPORTED model covers'),

    ('quality', f'{ROOT}/f5_pose_quality/posecheck_summary.csv', None,
     ['n_pockets', 'n_attempted', 'n', 'clash_median', 'strain_median', 'inter_mean',
      'inter_per_atom'], 'f5',
     'molecule-pooled (mean/median over molecules)',
     'docking-successful only; n < n_attempted by the PoseCheck timeout count'),
]


# Which sub-table each family's MAIN csv corresponds to (None = the family has only one).
MAIN_TABLE = {'sbdd': None, 'pose': 'rmsd', 'nci': 'nci_summary',
              'delta': None, 'quality': 'posecheck'}


def _status_cell(m, fam):
    """Status, qualified by sub-table membership.

    A model can be `ok` for a family and still contribute no row to that family's MAIN csv,
    because it lives in a sub-table -- `ours_noguide` is in F1's guidance-ablation block, novdw
    and head1_dock_342k in the physics-ablation block. Rendering a bare "ok" next to an empty
    row would read as a broken join, which is the one thing this table must never do.
    """
    st = model_registry.status(m, fam) or ''
    if st != 'ok':
        return st
    sec = model_registry.section(m, fam)
    if sec:
        return f'ok[{sec.replace("_ablation", "-abl")}]'
    # Same problem, different mechanism: a family can own several tables over overlapping
    # rosters, and `ok` for the family does not mean "in the MAIN one". head1_dock_342k is nci-ok
    # but lives in the IFP appendix; BInD is quality-ok but in neither quality table yet.
    main = MAIN_TABLE.get(fam)
    tables = (m.get('families', {}).get(fam) or {}).get('tables')
    if main and tables is not None and main not in tables:
        return f'ok[{",".join(tables) if tables else "no table"}]'
    return st


def load(path, keep=None):
    if not os.path.isfile(path):
        return {}, []
    with open(path, newline='', encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for r in rows:
        if keep and not keep(r):
            continue
        rid = (r.get('id') or '').strip()
        if rid:
            out[rid] = r
    cols = [c for c in (rows[0].keys() if rows else []) if c not in ('id',)]
    return out, cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--all_columns', action='store_true',
                    help='every column of every family instead of the headline set')
    ap.add_argument('--out', default=f'{ROOT}/master_table')
    ap.add_argument('--tiers', nargs='+',
                    default=['reference', 'core', 'extended', 'ablation'])
    a = ap.parse_args()

    tables = {}
    for fam, path, keep, cols, prefix, est, pop in SOURCES:
        data, allcols = load(path, keep)
        use = allcols if a.all_columns else cols
        tables[fam] = (data, [c for c in use if c in (allcols or use)], prefix, path, est, pop)
        print(f'  {fam:8s} {len(data):3d} rows  <- {path}')

    models = [m for m in model_registry.models() if m.get('tier') in a.tiers]
    models.sort(key=lambda m: (m.get('order', 10**6), m['id']))

    header = ['id', 'label', 'tier']
    for fam, (data, cols, prefix, _, _e, _p) in tables.items():
        header.append(f'{prefix}_status')
        header += [f'{prefix}_{c}' for c in cols]

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    csv_path, txt_path = a.out + '.csv', a.out + '.txt'
    with open(csv_path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for m in models:
            row = [m['id'], m['label'], m.get('tier')]
            for fam, (data, cols, prefix, _, _e, _p) in tables.items():
                st = _status_cell(m, fam)
                r = data.get(m['id'])
                row.append(st)
                row += [(r or {}).get(c, '') for c in cols]
            w.writerow(row)

    # ---- readable twin
    L = []
    def A(x=''):
        L.append(x)
    A('=' * 118)
    A('MASTER TABLE -- every family, one row per model, joined on the registry id')
    A('=' * 118)
    A('')
    A('A BLANK IS NOT A ZERO. Read the <family>_status column beside it:')
    A('  ok / not_measured / not_applicable / deferred / retired / (empty = not in that family)')
    A('The row axis is configs/models.json, not any one table, so a model measured nowhere still')
    A('appears here rather than silently vanishing.')
    A('')
    A('!! THE FAMILIES DO NOT SHARE AN ESTIMATOR OR AN n. Ranking a model by an f1 column against')
    A('   another model by an f4 column is meaningless, and no amount of shared formatting makes')
    A('   it valid. Every family restricts to DOCKING-SUCCESSFUL molecules -- that rule is shared')
    A('   -- but each instrument then loses its own molecules on top, so the realised n differs.')
    A('   Compare WITHIN a block. Across blocks, compare only the direction of a ranking.')
    A('')
    for fam, (data, cols, prefix, path, est, pop) in tables.items():
        A(f'  {prefix}  {fam}')
        A(f'        source     {path}')
        A(f'        estimator  {est}')
        A(f'        population {pop}')
    A('')
    A('Per-family gotchas, repeated here because this is where a reader is tempted to ignore them:')
    A('  f2  each engine uses a different box and search depth -- rank WITHIN a column.')
    A('  f3  PLIP counts atom pairs, ProLIF caps at one per (residue,type). Never sum/difference')
    A('      them, and never cross-cite the ProLIF columns (locked to prolif 2.2.0).')
    A('  f4  delta is ~-0.9 correlated with the absolute on-target score; read off_med beside it.')
    A('')
    A('=' * 118)
    for fam, (data, cols, prefix, _, est, pop) in tables.items():
        A('')
        A(f'{prefix.upper()}  {fam}')
        A(f'    estimator : {est}')
        A(f'    population: {pop}')
        A('-' * 118)
        # Column width follows the longest NAME in the block. Truncating to a fixed 13 made
        # rmsd_dock_vina_med and rmsd_dock_vinardo_med both print as "rmsd_dock_vin" -- two
        # different engines under one heading, in a table whose whole point is disambiguation.
        cw = max([len(c) for c in cols] + [8])
        A(f'{"model":26s} {"status":15s} ' + ' '.join(f'{c:>{cw}s}' for c in cols))
        for m in models:
            st = _status_cell(m, fam) or '-'
            r = data.get(m['id'])
            if r is None:
                A(f'{m["label"][:26]:26s} {st:15s} ' + ' '.join(f'{"--":>{cw}s}' for _ in cols))
                continue
            vals = [(r or {}).get(c, '') for c in cols]
            A(f'{m["label"][:26]:26s} {st:15s} '
              + ' '.join(f'{(v if v not in ("", None) else "--"):>{cw}s}' for v in vals))
    A('')
    A('=' * 118)
    A(f'Generated by scripts/build_master_table.py from configs/models.json + the five family CSVs.')
    A('=' * 118)
    open(txt_path, 'w').write('\n'.join(L) + '\n')

    dict_path = a.out + '_columns.csv'
    with open(dict_path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['column', 'family', 'prefix', 'source_file', 'estimator', 'population'])
        for c in ('id', 'label', 'tier'):
            w.writerow([c, '', '', 'configs/models.json', 'n/a (identity)', 'n/a'])
        for fam, (data, cols, prefix, path, est, pop) in tables.items():
            w.writerow([f'{prefix}_status', fam, prefix, 'configs/models.json',
                        'n/a (declared availability)', pop])
            for c in cols:
                w.writerow([f'{prefix}_{c}', fam, prefix, path, est, pop])

    print(f'\n-> {csv_path}   ({len(models)} models x {len(header)} columns)')
    print(f'-> {txt_path}')
    print(f'-> {dict_path}   (per-column estimator + population; a CSV cannot carry a second')
    print(f'                  header row, so the metadata lives here rather than nowhere)')
    _write_coverage(tables, models, a.out + '_coverage')
    return _self_check(csv_path)


def _write_coverage(tables, models, out):
    """WHICH BASELINE IS IN WHICH EXPERIMENT -- the paper-facing table.

    Separate from master_table on purpose. That one answers "what did each model score"; this one
    answers "who was even in the room", which is the question a reviewer asks first when the
    baseline set differs per experiment. Mixing the two buries it: 52 columns of numbers is where
    an uneven roster goes to hide.

    The cell is the realised n, not a tick, because participation is not binary -- PIDiff enters
    F1 with 850 molecules and TargetDiff with 9036, and a tick would call those the same thing.
    """
    N_COL = {'sbdd': 'n_mols', 'pose': 'rmsd_dock_vinardo_n', 'nci': 'n_plip',
             'delta': 'n_mol', 'quality': 'n'}
    TITLE = {'sbdd': 'F1 SBDD', 'pose': 'F2 RMSD', 'nci': 'F3 NCI',
             'delta': 'F4 Delta', 'quality': 'F5 PoseCheck'}
    # Appendix arms are registry-only (no CSV of their own); their coverage is the most uneven of
    # all, which is exactly why leaving them off would flatter the picture.
    APPX = [('PoseBusters', 'quality', 'posebusters'), ('IFP', 'nci', 'ifp')]

    def cell(m, fam):
        st = model_registry.status(m, fam)
        if st is None:
            return '.'
        if st == 'not_applicable':
            return 'n/a'
        if st == 'not_measured':
            return '-'
        if st == 'deferred':
            return 'def'
        if st == 'retired':
            return 'ret'
        data, cols, prefix, _p, _e, _pop = tables[fam]
        r = data.get(m['id'])
        if r is None:
            return 'sub'                      # measured, but rendered in a sub-table
        v = r.get(N_COL[fam], '')
        try:
            return str(int(float(v)))
        except (TypeError, ValueError):
            return 'Y'

    fams = list(tables)
    rows = []
    for m in models:
        rows.append([m['label'], m.get('tier', '')]
                    + [cell(m, f) for f in fams]
                    + ['Y' if model_registry.in_table(m, fam, tbl) else '-'
                       for _t, fam, tbl in APPX])

    header = ['model', 'tier'] + [TITLE[f] for f in fams] + [f'appx:{t}' for t, _, _ in APPX]
    with open(out + '.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)

    W = [max(len(str(r[i])) for r in rows + [header]) for i in range(len(header))]
    L = []
    def A(x=''):
        L.append(x)
    A('=' * (sum(W) + 2 * len(W)))
    A('BASELINE COVERAGE -- which model took part in which experiment')
    A('=' * (sum(W) + 2 * len(W)))
    A('')
    A('The cell is the number of MOLECULES that model contributed to that experiment, because')
    A('participation is not binary: PIDiff enters F1 with 850 molecules and TargetDiff with 9036.')
    A('')
    A('  <n>    took part, with that many molecules in the main table')
    A('  sub    measured, but reported in a sub-table (ablation block / appendix), not the main one')
    A('  -      NOT measured; re-measuring was out of scope, so the cell is empty by decision')
    A('  n/a    structurally impossible -- e.g. PIDiff has 8.5 mols/pocket vs the 20 F4 requires')
    A('  def    data exists on disk, wiring the row was deferred')
    A('  ret    measured and kept, deliberately withdrawn from reporting (2026-08-05)')
    A('  .      the model is not part of that family at all')
    A('')
    A('n is NOT comparable across columns: every experiment restricts to docking-successful')
    A('molecules, but each instrument then loses its own (PoseCheck timeouts, engine failures),')
    A('and F4 is on the 95-pocket common set while the rest are on 100. See master_table.txt.')
    A('')
    A('  ' + '  '.join(f'{h:<{W[i]}s}' if i < 2 else f'{h:>{W[i]}s}'
                       for i, h in enumerate(header)))
    A('  ' + '  '.join('-' * W[i] for i in range(len(header))))
    prev = None
    for m, r in zip(models, rows):
        if prev is not None and r[1] != prev:
            A('  ' + '  '.join('.' * W[i] for i in range(len(header))))
        prev = r[1]
        A('  ' + '  '.join(f'{str(v):<{W[i]}s}' if i < 2 else f'{str(v):>{W[i]}s}'
                           for i, v in enumerate(r)))
    A('')
    counted = []
    for i, f in enumerate(fams):
        n = sum(1 for r in rows if r[2 + i] not in ('-', 'n/a', 'def', 'ret', '.', 'sub'))
        counted.append(f'{TITLE[f]}={n}')
    A('  models in each main table:  ' + '   '.join(counted))
    A('')
    open(out + '.txt', 'w').write('\n'.join(L) + '\n')
    print(f'-> {out}.csv / {out}.txt   (baseline coverage matrix)')


def _self_check(csv_path):
    """Status and emptiness must agree in every cell.

    A bare `ok` beside an all-empty block means the join silently failed; a `not_measured` beside
    real numbers means the registry is lying about what exists. Either way the table is worse than
    no table, so this refuses to pass quietly.
    """
    with open(csv_path, newline='', encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))
    prefixes = sorted({k.split('_')[0] for k in (rows[0] if rows else {}) if k.startswith('f')})
    bad = []
    for r in rows:
        for p in prefixes:
            st = r.get(f'{p}_status', '')
            filled = [v for k, v in r.items()
                      if k.startswith(p + '_') and not k.endswith('_status') and v not in ('', None)]
            if st == 'ok' and not filled:
                bad.append(f"{r['label']} / {p}: status 'ok' but every cell is empty")
            if st in ('not_measured', 'not_applicable', 'deferred', 'retired', '') and filled:
                bad.append(f"{r['label']} / {p}: status {st!r} but {len(filled)} cell(s) have values")
    if bad:
        print(f'\nSELF-CHECK FAILED ({len(bad)}):')
        for b in bad:
            print('  ' + b)
        return 1
    print(f'self-check OK: status and emptiness agree in all '
          f'{len(rows) * len(prefixes)} model x family cells')
    return 0


if __name__ == '__main__':
    sys.exit(main())
