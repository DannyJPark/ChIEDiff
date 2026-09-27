#!/usr/bin/env python3
"""Regenerate results/comparison/f2_pose_fidelity/rmsd_summary.txt from master.csv + self_redock_v4 / self_redock_min_v5/.

All pose-fidelity models (registry order) + a REFERENCE (self-redock ceiling) row in sections 1/2.
Docking-successful molecules only; pocket-level bootstrap CI. Idempotent -- re-run any time.

Three engine columns: smina-Vinardo, gnina (CNN), AutoDock Vina (meeko). The smina *Vina* column
was dropped 2026-08-05 -- smina's default scoring IS Vina 1.1.2, so it duplicated the AutoDock
Vina column's functional form while looking like independent evidence. The one place it survives
is section 4, where the paired (Vinardo - Vina) contrast IS the measurement: same binary, same
search, same box, same seed, same atom typing, scoring function the only variable. Raw values
stay in master.csv (rmsd_dock_smina / rmsd_min_smina).

Of the columns shown, AutoDock Vina (meeko) is Vina 1.2.2 and gnina searches with a Vina-family
score (CNN only rescores the final pose), so both still share the functional form our physics loss
derives from -- that circularity (defect D8) is exactly what the Vinardo column tests.

Reference dirs (2026-08), kept in step with build_redock_comparison.py so both files quote the
same ceiling:  self_redock_v4  and  self_redock_min_v5.
  _v2 = the 2026-07-28 sanitisation fix;  _v3 = _v2 + the Vinardo arm;
  _v4 = _v3 + the AutoDock Vina dock affinity recorded on the pose;
  _min_v5 = _min_v4 + the AutoDock Vina MINIMIZE affinity, which needed v.score() after
            optimize() -- write_pose() emits no 'REMARK VINA RESULT' to parse and energies()
            returns []. RMSD columns are byte-identical across v3/v4/v5 (seeded, deterministic);
            only affinity columns were added, so the ceiling RMSDs did not move.
The original `self_redock/` and `self_redock_min/` are the PRE-FIX instrumentation in which
66/100 ceiling pockets fell back to skeleton isomorphism -- a more permissive instrument than the
model rows it bounds -- and must not be read again.
"""
import csv, glob, json, os
import numpy as np
from collections import defaultdict

SELF_DOCK_DIR = 'results/reference_protocol/self_redock_v4'
SELF_MIN_DIR  = 'results/reference_protocol/self_redock_min_v5'

def _boot(pairs, B=3000):
    if not pairs: return None
    rng = np.random.default_rng(0); by = defaultdict(list)
    for p, v in pairs: by[p].append(v)
    ks = list(by); a = {k: np.array(by[k]) for k in ks}
    allv = np.concatenate([a[k] for k in ks]); pt = np.median(allv)
    idx = rng.integers(0, len(ks), size=(B, len(ks)))
    o = [np.median(np.concatenate([a[ks[i]] for i in idx[b]])) for b in range(B)]
    return pt, np.percentile(o, 2.5), np.percentile(o, 97.5), len(pairs), 100*np.mean(allv<2), 100*np.mean(allv<1)

def cell(r): return f'{r[0]:.2f} [{r[1]:.2f},{r[2]:.2f}]' if r else '       --       '


def paired_delta(rows, tag, col_a, col_b, B=3000):
    """Median of (b - a) over molecules where BOTH engines produced an RMSD, with a pocket-level
    bootstrap CI.

    This is the statistic to quote, NOT the ratio of the two column medians. A ratio inflates
    whenever the denominator is small: our smina minimize median is 0.23 A, so a +0.07 A shift
    reads as "1.39x" while the same +0.07 A on a 0.63 A baseline reads as "1.11x" -- an artefact
    of the denominator, not a difference in behaviour. Pairing also removes molecule-to-molecule
    variance, which is far larger than the effect being measured."""
    pairs = []
    for r in rows:
        if r['model'] != tag:
            continue
        a, b = r.get(col_a), r.get(col_b)
        if a and b:
            pairs.append((r['pocket_idx'], float(b) - float(a)))
    return _boot(pairs, B=B)


def dcell(r): return f'{r[0]:+.3f} [{r[1]:+.3f},{r[2]:+.3f}]' if r else '         --        '


def frac_under(rows, tag, col, thr=2.0, B=3000):
    """% of the model's DOCKING-SUCCESSFUL molecules whose RMSD is < thr, with a pocket-level
    bootstrap CI, plus the coverage that produced it.

    The denominator is EVERY docking-successful molecule of the model, not just the ones this
    engine returned a pose for. A molecule the engine could not measure counts as a failure,
    never as a silent exclusion -- otherwise a model whose hardest molecules are the unmeasurable
    ones gets its success rate computed on the easy remainder. Where coverage is 100% the two
    denominators coincide; where it is not (notably the AutoDock Vina arm, which was only run to
    completion on 5 models) the difference is exactly the bias this guards against.

    -> (pct, lo, hi, n_measured, n_docked) or None
    """
    by = defaultdict(lambda: [0, 0])          # pocket -> [hits, docked]
    n_meas = n_dock = 0
    for r in rows:
        if r['model'] != tag:
            continue
        p = r['pocket_idx']
        by[p][1] += 1
        n_dock += 1
        v = r.get(col)
        if v:
            n_meas += 1
            if float(v) < thr:
                by[p][0] += 1
    if not n_dock:
        return None
    ks = list(by)
    hits = np.array([by[k][0] for k in ks], dtype=float)
    tot = np.array([by[k][1] for k in ks], dtype=float)
    pct = 100.0 * hits.sum() / tot.sum()
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(ks), size=(B, len(ks)))
    o = 100.0 * hits[idx].sum(axis=1) / tot[idx].sum(axis=1)
    return pct, float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5)), n_meas, n_dock


def frac_under_ref(data, col, thr=2.0, B=3000):
    """Reference-row twin of frac_under(): % of the 100 reference ligands under thr.
    One molecule per pocket, so the pocket bootstrap is just a bootstrap over the 100 rows."""
    vals, n_dock = [], 0
    for r in data:
        n_dock += 1
        v = r.get(col)
        vals.append(1.0 if (v and float(v) < thr) else 0.0)
    if not n_dock:
        return None
    a = np.asarray(vals)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(a), size=(B, len(a)))
    o = 100.0 * a[idx].mean(axis=1)
    n_meas = sum(1 for r in data if r.get(col))
    return (100.0 * a.mean(), float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5)),
            n_meas, n_dock)


def pcell(r):
    """Never print a bare rate the reader could mistake for a model property.

    A cell whose engine measured nothing is not "0% of molecules docked within 2 A" -- it is
    "this arm was not run". Printing 0.0% there would read as the model failing every molecule.
    A partially covered cell is a valid LOWER BOUND under the strict denominator, but it mixes
    pose quality with non-measurement, so it is flagged rather than left to be read at face value.
    """
    if not r:
        return '        --       '
    pct, lo, hi, meas, dock = r
    if meas == 0:
        return '   not run       '
    mark = '' if meas >= dock else '†'          # dagger = incomplete coverage
    return f'{pct:5.1f}% [{lo:4.1f},{hi:4.1f}]{mark}'

def _rows_from(pattern):
    return [r for f in sorted(glob.glob(pattern)) for r in csv.DictReader(open(f))]

def main():
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    import model_registry                                    # noqa: E402
    reg = model_registry.load_as(model_registry.DEFAULT_PATH, 'pose')['models']
    # One label per model, from configs/models.json. The private dict this replaces disagreed
    # with build_redock_comparison.py's copy for the SAME model in the SAME family.
    labels = model_registry.labels_by('pose')
    order = [m['tag'] for m in reg]
    rows = [r for r in csv.DictReader(open('results/pose_fidelity/master.csv')) if r['orig_docked']=='1']
    def bm(t, col): return _boot([(r['pocket_idx'], float(r[col])) for r in rows if r['model']==t and r.get(col)])
    sd = _rows_from(f'{SELF_DOCK_DIR}/pocket*.csv')
    sm = _rows_from(f'{SELF_MIN_DIR}/pocket*.csv')
    def br(data, col): return _boot([(r['pocket_idx'], float(r[col])) for r in data if r.get(col)])

    W=130; L=[]; A=L.append
    A('='*W); A('POSE-FIDELITY RMSD SUMMARY  —  generated pose  vs  re-docked / minimized pose'); A('='*W); A('')
    A('Metric: non-superposed, symmetry-aware heavy-atom RMSD (rdMolAlign.CalcRMS). Angstrom, lower=better.')
    A('  GetBestRMS (superposing) is deliberately NOT used -- this measures in-pocket displacement.')
    A('  Atom map exact: meeko REMARK-SMILES (0 fallback / 0 Jensen violations); smina/vinardo/gnina round-trip')
    A('  SDF with a <0.6% skeleton-isomorphism fallback (baselines only).')
    A('Population: DOCKING-SUCCESSFUL molecules only (fragmented excluded, reported separately).')
    A('CI: 95% POCKET-level bootstrap. n = molecules (Reference row: n = pockets = 100).')
    A('')
    A('The REFERENCE row = the CrossDocked reference ligand self-(re)processed with the IDENTICAL')
    A('protocol (same box, exhaustiveness, RMSD method) -- only the INPUT differs. It is the ceiling.')
    A('NOTE the reference is a DOCKED/MINIMIZED pose, not an experimental crystal (64/100 cross-docked).')
    A('')
    A('WHY THE smina (Vina) COLUMN WAS REMOVED (2026-08-05, user decision)')
    A('  smina`s default scoring IS Vina 1.1.2, so that column measured the same functional form as')
    A('  the AutoDock Vina (meeko) column already shown -- two implementations of one scoring')
    A('  function, reported as if they were independent evidence. It is dropped from every table')
    A('  here. It survives in ONE place only: section 4, where the paired (Vinardo - Vina) contrast')
    A('  IS the measurement and deleting the Vina arm would destroy the single-variable control.')
    A('  Raw values remain in results/pose_fidelity/master.csv (rmsd_dock_smina, rmsd_min_smina).')
    A('')
    A('WHY THE VINARDO COLUMN IS THE ONE THAT SETTLES D8')
    A('  AutoDock Vina(meeko) = Vina 1.2.2, and gnina searches with a Vina-family score and applies')
    A('  the CNN only as a rescore of the final pose. Both therefore share the functional form our')
    A('  physics loss is derived from, so a low RMSD in them is partly circular.')
    A('  smina-Vinardo changes the SCORING FUNCTION AND NOTHING ELSE -- same binary, same Monte-Carlo')
    A('  search, same --autobox_ligand box, same seed 42, same exhaustiveness 8, same atom typing.')
    A('  Vinardo drops Vina`s gauss2 term, refits steric/hydrophobic/h-bond, and sets the rotatable-')
    A('  bond penalty weight to 0 (Vina: 1.923). Read this column as the independence test.')
    A('  Caveat: Vinardo is still an empirical function fit on similar data, sharing Vina`s atom typing')
    A('  and rigid-receptor assumption. It is an independent SCORING FUNCTION, not independent physics.')
    A('')
    A('COLUMN ORDER is AutoDock Vina | smina-Vinardo | gnina, matching redock_comparison.txt.')
    A('The two files publish the SAME redock and minimize RMSD numbers; they used to differ only')
    A('in column order, so reading position 1 in each gave 1.39 vs 0.74 for one model and looked')
    A('like disagreement. Unified 2026-08-21. Keep them in step.')
    A('')
    A('Boxes/exhaustiveness differ by engine -> absolute values NOT interchangeable, ranking is:')
    A('  vinardo/gnina --autobox_ligand <crystal>  exh 8   |   AutoDock Vina(meeko) ref 20A cube exh 16')
    A('')
    A('='*W); A('1. RE-DOCKED (full --dock re-search)   [pose-fidelity headline]'); A('='*W); A('')
    A(f'{"row":20s} {"AutoDock Vina (meeko)":>22s}   {"smina (VINARDO)":>18s} {"gnina (CNN)":>18s} {"n":>7s}')
    A('-'*W)
    rw,rg,rv = br(sd,'rmsd_vinardo'),br(sd,'rmsd_gnina'),br(sd,'rmsd_vina_meeko')
    A(f'{"REFERENCE (ceiling)":20s} {cell(rv):>22s}   {cell(rw):>18s} {cell(rg):>18s}   {"100":>5s}')
    A('-'*W)
    for t in order:
        w=bm(t,'rmsd_dock_vinardo'); g=bm(t,'rmsd_dock_gnina'); v=bm(t,'vm_rmsd_gen_dock')
        star=' *' if (v and w and v[3] < 0.7*w[3]) else '  '
        A(f'{labels.get(t,t):20s} {cell(v):>22s}{star} {cell(w):>18s} {cell(g):>18s}   {(str(w[3]) if w else "0"):>5s}')
    A('')
    A('* AutoDock Vina column: partial sample (outside the resume scope), shown for reference only.')
    A('')
    A('='*W); A('2. LOCALLY MINIMIZED (--minimize, stays put)   [is it already a local min?]'); A('='*W); A('')
    A('Box- and search-independent: --minimize relaxes IN PLACE, so this is the cleanest form of the')
    A('question "is the generated pose already at THIS scoring function`s optimum?". A pose that is a')
    A('Vina local minimum but not a Vinardo one shows up in section 4 as a positive paired shift.')
    A('')
    A(f'{"row":20s} {"AutoDock Vina (meeko)":>22s} {"smina (VINARDO)":>18s} {"gnina (CNN)":>18s} {"n":>7s}')
    A('-'*W)
    rmw,rmg,rmv = (br(sm,'min_rmsd_vinardo'),br(sm,'min_rmsd_gnina'),br(sm,'min_rmsd_vina_meeko'))
    A(f'{"REFERENCE (ceiling)":20s} {cell(rmv):>22s} {cell(rmw):>18s} {cell(rmg):>18s}   {"100":>5s}')
    A('-'*W)
    for t in order:
        w=bm(t,'rmsd_min_vinardo'); g=bm(t,'rmsd_min_gnina'); v=bm(t,'vm_rmsd_gen_min')
        n = w[3] if w else (g[3] if g else (v[3] if v else 0))
        A(f'{labels.get(t,t):20s} {cell(v):>22s} {cell(w):>18s} {cell(g):>18s}   {str(n):>5s}')
    A('')
    A('The Vinardo minimize column is complete for every model. The gnina column was never run for')
    A('pignet_fixed / pgdiff_fixed, so those show "--" -- left out by decision, not by failure.')
    A('')
    A('='*W)
    A('3. SUCCESS RATE: % of molecules with RMSD < 2 A')
    A('='*W); A('')
    A('DENOMINATOR = every DOCKING-SUCCESSFUL molecule of that model, not just the ones the engine')
    A('managed to measure. A molecule the engine returned no pose for counts as a FAILURE here.')
    A('That is the conservative choice: the unmeasurable molecules are the pathological ones, so')
    A('dropping them from the denominator would compute a success rate on the easy remainder and')
    A('flatter whichever model produces more of them.')
    A('Where an engine covered 100% of the population the two denominators coincide; the cells')
    A('where they do not are listed under the table, with the coverage that produced them.')
    A('')
    A('2 A is the conventional redocking-success threshold. CI = 95% pocket-level bootstrap.')
    A('')
    for title, cols, refdata, refcols in (
        ('3a. RE-DOCKED (--dock)',
         ('vm_rmsd_gen_dock', 'rmsd_dock_vinardo', 'rmsd_dock_gnina'),
         sd, ('rmsd_vina_meeko', 'rmsd_vinardo', 'rmsd_gnina')),
        ('3b. LOCALLY MINIMIZED (--minimize)',
         ('vm_rmsd_gen_min', 'rmsd_min_vinardo', 'rmsd_min_gnina'),
         sm, ('min_rmsd_vina_meeko', 'min_rmsd_vinardo', 'min_rmsd_gnina')),
    ):
        A(title)
        A(f'{"row":20s} {"AutoDock Vina":>19s} {"smina (VINARDO)":>19s} '
          f'{"gnina (CNN)":>19s} {"n docked":>9s}')
        A('-'*W)
        rcells = [pcell(frac_under_ref(refdata, c)) for c in refcols]
        A(f'{"REFERENCE (ceiling)":20s} ' + ' '.join(f'{c:>19s}' for c in rcells) + f' {100:9d}')
        A('-'*W)
        partial = []
        for t in order:
            cells, nd = [], 0
            for c in cols:
                r = frac_under(rows, t, c)
                cells.append(pcell(r))
                if r:
                    nd = r[4]
                    if r[3] < r[4]:
                        partial.append((labels.get(t, t), c, r[3], r[4]))
            A(f'{labels.get(t,t):20s} ' + ' '.join(f'{c:>19s}' for c in cells) + f' {nd:9d}')
        A('-'*W)
        if partial:
            A('† = the engine measured FEWER molecules than the model has docking-successful, so the')
            A('  shortfall is counted as failure and the number is a LOWER BOUND on pose quality,')
            A('  not a clean estimate of it. "not run" = that arm produced nothing for this model;')
            A('  it is NOT a 0% success rate. Do not rank a flagged cell against an unflagged one.')
            for lab, c, meas, dock in partial:
                # show the raw shortfall: 8339/8340 rounds to "100.0% covered", which reads as
                # complete and would make the dagger look like a mistake
                A(f'    {lab:20s} {c:20s} {meas:6d}/{dock:<6d}  '
                  f'({dock - meas:d} unmeasured, {100*meas/dock:.2f}% covered)')
        else:
            A('Every engine covered 100% of the docking-successful population in this block.')
        A('')
    A('='*W)
    A('4. THE D8 TEST: paired per-molecule (Vinardo - Vina) shift, A')
    A('='*W); A('')
    A('THIS is the one place the smina (Vina) arm is still used, deliberately. Everywhere else it was')
    A('removed as a duplicate of AutoDock Vina, but here the (Vinardo - Vina) contrast IS the')
    A('measurement: dropping the Vina arm would remove the control, not a redundant column.')
    A('')
    A('Same molecule, same box, same seed, same search -- only the scoring function differs. So a')
    A('positive shift means "this pose sits closer to a Vina optimum than to a Vinardo one", i.e.')
    A('exactly what D8 (Goodhart on Vina) predicts should be LARGER for models trained with a')
    A('Vina-derived physics loss than for models that never saw one.')
    A('')
    A('Quote THIS, not the ratio of the two column medians. A ratio explodes when the denominator')
    A('is small: +0.07 A on our 0.23 A minimize median reads as "1.39x", while the identical')
    A('+0.07 A on TargetDiff`s 0.63 A reads as "1.11x". That is the denominator talking, not the')
    A('models. Pairing also cancels molecule-to-molecule variance, which dwarfs the effect.')
    A('')
    A(f'{"row":20s} {"dock: Vnd-Vina":>26s} {"minimize: Vnd-Vina":>26s} {"n(dock)":>9s}')
    A('-'*W)
    for t in order:
        dd = paired_delta(rows, t, 'rmsd_dock_smina', 'rmsd_dock_vinardo')
        dm = paired_delta(rows, t, 'rmsd_min_smina', 'rmsd_min_vinardo')
        A(f'{labels.get(t,t):20s} {dcell(dd):>26s} {dcell(dm):>26s} '
          f'{(str(dd[3]) if dd else "0"):>9s}')
    A('-'*W)
    A('')
    A('Read the magnitudes before the ranking: every shift here is a few HUNDREDTHS of an Angstrom,')
    A('against pose differences between models of 0.6-3.7 A. Whatever Vina tilt exists is real but')
    A('two orders of magnitude smaller than the effect it would need to explain.')
    A('')
    A('='*W); A('KEY: why the Reference re-dock (sec.1) is HIGH while its minimize (sec.2) is ~0'); A('='*W); A('')
    A('The CrossDocked "reference" ligand is NOT an experimental crystal pose -- it is a DOCKED/MINIMIZED')
    A('pose (filenames *_tt_docked_/_tt_min_), 64/100 CROSS-docked (receptor & ligand from different PDB).')
    A('  minimize (sec.2): --minimize relaxes IN PLACE. Under gnina the reference sits at 0.00 A because')
    A('    the CrossDocked pose was produced by Vina-family minimisation, so it IS that optimum already.')
    A('    Under VINARDO it moves 0.29 A -- the floor any Vina-optimised pose pays for the swap.')
    A('  dock (sec.1 Reference 3.63 A Vinardo): --dock DISCARDS the input coordinates and re-searches the')
    A('    whole box; the scoring function`s global optimum is a flipped/relocated pose, not this one.')
    A('')
    A('READ THE TOP ROW BEFORE THE MODEL ROWS. Every model re-docks CLOSER to its own generated pose')
    A('than the reference ligand re-docks to its own pose (Vinardo: ours 0.74 A / 64.9% under 2 A, vs')
    A('reference 3.63 A / 31.0%). That is not the models beating a crystal structure. It means their')
    A('poses already sit at the scoring function`s global optimum, which the reference does not --')
    A('the reference is a real binding mode, the generated pose is a scoring-function optimum, and this')
    A('metric rewards the second. Quote a low dock-RMSD as "at the docking optimum", never as')
    A('"matches the crystal pose".')
    A('')
    A('Whether that optimum-seeking is Goodhart (Vina memorised) or genuine physics is what the VINARDO')
    A('column answers, and the paired test in section 4 answers it quantitatively: the tilt towards Vina')
    A('is real but ~0.05 A, roughly 2% of the 2.6 A gap it would have to explain.')
    A('')
    A('='*W)
    A(f'Source: results/pose_fidelity/master.csv ; {SELF_DOCK_DIR}/pocket*.csv ; {SELF_MIN_DIR}/pocket*.csv')
    A('='*W)
    open('results/comparison/f2_pose_fidelity/rmsd_summary.txt','w').write('\n'.join(L)+'\n')
    print('\n'.join(L))
    _write_csv(order, labels, bm, br, sd, sm)


def _write_csv(order, labels, bm, br, sd, sm,
               out='results/comparison/f2_pose_fidelity/rmsd_summary.csv'):
    """Machine-readable twin of the report, one row per model.

    The report is for reading; this is for joining. It carries the registry `id` as the join key
    because a LABEL IS NOT A JOIN KEY -- that is the mistake this whole reorganisation existed to
    undo, and a downstream integrated table must not reintroduce it.

    _boot() returns (median, ci_lo, ci_hi, n, pct<2A, pct<1A), so every engine/mode contributes
    those six numbers. A cell the engine never produced is left EMPTY, never 0: "not measured"
    and "measured as zero" are different facts.
    """
    import csv as _csv, sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    import model_registry                                                # noqa: E402

    tag2id = {(m.get('ids') or {}).get('eval_out'): m['id']
              for m in model_registry.models() if (m.get('ids') or {}).get('eval_out')}
    # column order matches the report: AutoDock Vina | smina-Vinardo | gnina
    ARMS = [('dock', 'vina',    'vm_rmsd_gen_dock',  'rmsd_vina_meeko'),
            ('dock', 'vinardo', 'rmsd_dock_vinardo', 'rmsd_vinardo'),
            ('dock', 'gnina',   'rmsd_dock_gnina',   'rmsd_gnina'),
            ('min',  'vina',    'vm_rmsd_gen_min',   'min_rmsd_vina_meeko'),
            ('min',  'vinardo', 'rmsd_min_vinardo',  'min_rmsd_vinardo'),
            ('min',  'gnina',   'rmsd_min_gnina',    'min_rmsd_gnina')]
    STATS = ['med', 'lo', 'hi', 'n', 'pct_lt2', 'pct_lt1']

    header = ['id', 'label', 'row_kind']
    for mode, eng, _, _ in ARMS:
        header += [f'rmsd_{mode}_{eng}_{s}' for s in STATS]

    def emit(w, rid, label, kind, getter):
        row = [rid, label, kind]
        for mode, eng, mcol, rcol in ARMS:
            r = getter(mode, mcol, rcol)
            row += [round(float(r[i]), 4) for i in range(6)] if r else [''] * 6
        w.writerow(row)

    _os.makedirs(_os.path.dirname(_os.path.abspath(out)), exist_ok=True)
    with open(out, 'w', newline='') as fh:
        w = _csv.writer(fh)
        w.writerow(header)
        # The reference row is the ligand self-redocked/self-minimized, so it reads from the
        # self_redock (sd) / self_min (sm) tables, not from master.csv like the model rows.
        emit(w, 'reference', 'Reference (native)', 'ceiling',
             lambda mode, mcol, rcol: br(sd if mode == 'dock' else sm, rcol))
        for t in order:
            emit(w, tag2id.get(t, t), labels.get(t, t), 'model',
                 lambda mode, mcol, rcol, _t=t: bm(_t, mcol))
    print(f'-> {out}')

if __name__ == '__main__':
    main()
