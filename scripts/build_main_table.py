#!/usr/bin/env python3
"""Main table: binding affinity across THREE scoring functions + molecular properties.

The F1 table next door (comparison_tables.txt) reports affinity from one engine only -- AutoDock
Vina, read out of each model's own sampling .pt. This table puts three scoring functions side by
side, each in two states, and adds the gap between those states:

    Vina  | Vinardo | gnina        each as  Score (pose as generated) / Dock (full redock) / |gap|
    HA | QED | SA | Div

`Score` is the molecule's own generated pose, rescored in place (--score_only, no box). `Dock` is a
full re-search. |gap| = per-molecule abs(Score - Dock), then averaged -- NOT the difference of the
two aggregates, which carries the opposite sign on this dataset (scripts/analyze_score_vs_dock.py).

POPULATION: a molecule counts only if ALL SIX affinities exist. Every column, including QED / SA /
Div / HA, is computed on that same intersected set, so no column has a denominator of its own.

Reads:
    <each model's sbdd .pt>                             Vina Score/Dock, QED, SA, fingerprints
    eval_out/<tag>/{smina_vinardo,gnina}/pocket*_{score,dock}.sdf     SD tag minimizedAffinity
    eval_out/<tag>/docked_names.txt                     the docking-successful population
    results/reference_protocol/...                      the Reference row (see reference_row)

Writes:
    results/comparison/f1_sbdd/main_table/main_table.txt
    results/comparison/f1_sbdd/main_table/main_table.csv

Usage:
    python scripts/build_main_table.py
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_registry                                                        # noqa: E402
from build_comparison_tables import agg, fmt, get_vina, pocket_ha, pocket_div  # noqa: E402
from build_redock_comparison import read_dock_tags                           # noqa: E402
from eval_export_sdf import load_grouped                                     # noqa: E402

EVAL_OUT = 'eval_out'

# The rows, by registry id. by_id raises KeyError on a typo, which is what we want -- a silently
# dropped row would read as "that model has no data" rather than "that id is wrong".
#
# `novdw` (Ours (Guide Only)) is deliberately NOT here: it is an ablation arm, and this table is
# the paper's main comparison. It remains in the F1 table next door, which is where its ablation
# reading belongs.
ROW_IDS =['reference', 'pocket2mol', 'targetdiff', 'ipdiff', 'alidiff', 'molcraft',
           'pidiff_retrain', 'kgdiff', 'ours_noguide', 'ours_vina']

# Our own rows are PINNED to the bottom in this order regardless of HA, so the reader always finds
# them in the same place and the baseline block above them stays a clean HA ranking. Without the
# pin, Ours (Vina Only) (HA 69.3) would sort into the middle of the baselines and split the two
# arms of our own model across the table.
OURS_LAST = ['ours_noguide', 'ours_vina']

# engine key -> the eval_out/<tag>/ subdirectory holding its poses.
# 'smina_vinardo' NOT 'smina': smina's DEFAULT scoring function IS AutoDock Vina 1.1.2, so a
# smina-default column reproduces the gnina affinity column to ~0.014 kcal/mol (measured, see
# f2_pose_fidelity/redock_comparison.txt section 3) -- three columns, two independent functions.
# It is also absent for IPDiff and Ours (Vina Only). Vinardo is a genuinely different functional
# form with complete coverage, at the cost of being on its own scale.
ENGINES = [('vinardo', 'smina_vinardo'), ('gnina', 'gnina')]

# gnina's CNN affinity head, carried as its own columns.
#
# WHY IT IS HERE. The three affinity columns are less independent than "Vina | Vinardo | gnina"
# suggests: gnina's minimizedAffinity IS the Vina 1.1.2 functional form, and measured on these
# molecules it tracks our Vina Score column at Spearman +0.94..+0.96 (median |diff| 0.44-0.52
# kcal/mol). Vinardo is not much further away (gnina S vs Vinardo S, r +0.95..+0.97). CNNaffinity
# is the one readout in this table that does NOT come from a Vina-family closed-form expression --
# it is a 3-D CNN regression trained on experimental binding constants -- and it agrees with the
# rest far less (|rho| 0.57..0.81), which is exactly why it earns a column.
#
# It is NOT a unit conversion of minimizedAffinity. A conversion would force
# minimizedAffinity == -1.364 * CNNaffinity (dG = -2.303*R*T*pK at 298 K) with zero residual;
# measured residual sd is 5.7-6.5 kcal/mol, and no linear map does better (best-fit residual sd
# 5.5-6.5). Different quantity, different scale, OPPOSITE direction.
#
# (record key, engine subdir, mode, SD tag / reference CSV column)
CNN_COLS = [('cnnaff_score', 'gnina', 'score', 'CNNaffinity', 'gnina_CNNaffinity'),
            ('cnnaff_dock', 'gnina', 'dock', 'CNNaffinity', 'gnina_CNNaffinity')]

# |E| this large is not an affinity, it is a docking grid's out-of-the-box penalty (~1e6 kcal/mol
# per atom outside the map). Same threshold as build_redock_comparison / build_reference_affinity.
PENALTY = 1e3

# Reference-ligand protocol. Parameters are IDENTICAL to the model runs -- verified in
# scripts/self_redock.py:53-56 against scripts/sbatch_dock_worklist.sh:110 (score is --score_only
# and takes no box; dock is --autobox_ligand <crystal> --exhaustiveness 8 --num_modes 1 --seed 42).
REF_SCORE_DIR = 'results/reference_protocol/self_reference_affinity_v2'
REF_DOCK_DIR = 'results/reference_protocol/self_redock_v4'

# (record key, display label, decimal places). Order IS the table's column order.
GROUPS = [
    ('vina_score', 'Vina S', 3), ('vina_dock', 'Vina D', 3), ('vina_gap', '|dV|', 3),
    ('vinardo_score', 'Vinardo S', 3), ('vinardo_dock', 'Vinardo D', 3), ('vinardo_gap', '|dS|', 3),
    ('gnina_score', 'gnina S', 3), ('cnnaff_score', 'CNNaff S', 3),
    ('gnina_dock', 'gnina D', 3), ('cnnaff_dock', 'CNNaff D', 3), ('cnnaff_gap', '|dG|cnn', 3),
    ('ha', 'HA', 1), ('qed', 'QED', 3), ('sa', 'SA', 3), ('div', 'Div', 3),
]
# Pooled over molecules; the other two are per-pocket first (see aggregate_row).
POOLED = [k for k, _, _ in GROUPS if k not in ('ha', 'div')]

# The six kcal/mol affinities. ONLY these get the grid-penalty filter and feed max|E|: CNNaffinity
# is a bounded pK regression that cannot express an out-of-the-box penalty, so screening it against
# a 1e3 kcal/mol threshold would be a category error.
AFFINITY_KEYS = ['vina_score', 'vina_dock'] + [f'{n}_{m}' for n, _ in ENGINES
                                               for m in ('score', 'dock')]


def _raw(x):
    """float-or-None, WITHOUT the penalty filter. Use when a penalty has to be counted."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _f(x):
    v = _raw(x)
    return None if v is None or abs(v) >= PENALTY else v


def coord_hash(mol):
    """Content key for a generated pose, stable across .pt files that hold the same molecules.

    Docking results live under the 'vina' key and do NOT overwrite 'mol', so the generated
    conformer is byte-identical between a model's raw export and its docked twin.
    """
    if mol is None or mol.GetNumConformers() == 0:
        return None
    return hash(np.round(mol.GetConformer().GetPositions(), 3).tobytes())


def keyed_entries(pt_path):
    """-> {p<NNN>_m<MMMM>: entry} using the SAME indexing eval_export_sdf.py wrote the SDFs with.

    load_grouped places flat layouts at their CANONICAL data_id, and `mi` advances even for
    molecules that are skipped, so the key encodes position in the .pt. That is exactly what the
    SDF titles mean, which is why this must not be re-derived by hand.
    """
    out = {}
    for pi, pocket in enumerate(load_grouped(pt_path)):
        for mi, e in enumerate(pocket):
            mol = e.get('mol')
            if mol is None or mol.GetNumConformers() == 0:
                continue
            out[f'p{pi:03d}_m{mi:04d}'] = e
    return out


def remap_by_pose(key_entries, metric_pt):
    """Re-point SDF-space keys at a DIFFERENT .pt's entries, matching on the generated pose.

    Needed because an SDF title indexes into the .pt the SDFs were EXPORTED from, which is not
    always the .pt the F1 metrics live in. `ours_noguide` is the live case: its eval_out/ SDFs came
    from noguide_grouped.pt (which carries no vina and no chem_results at all) while Vina/QED/SA
    live in noguide_merged_vina_docked.pt, and the two number their molecules differently -- only
    1942 of 2145 keys are even shared, and barely 60% of those are the same molecule. Joining on
    the title would pair the wrong molecule for about 40% of that row.

    Raises rather than returning a partial map: a silent 60%-correct join is the exact failure this
    function exists to prevent.
    """
    by_hash = {}
    for e in keyed_entries(metric_pt).values():
        h = coord_hash(e.get('mol'))
        if h is None:
            continue
        by_hash.setdefault(h, []).append(e)

    out, missing, ambiguous = {}, 0, 0
    for k, e in key_entries.items():
        cand = by_hash.get(coord_hash(e.get('mol')), [])
        if not cand:
            missing += 1
        elif len(cand) > 1:
            ambiguous += 1
        else:
            out[k] = cand[0]
    if missing or ambiguous:
        raise SystemExit(
            f'[fatal] pose remap onto {metric_pt} is not 1:1 '
            f'({missing} unmatched, {ambiguous} ambiguous of {len(key_entries)}). '
            f'Refusing to emit a table built on a partial join.')
    return out


def engine_tags(tag, engine_dir, mode, sd_tag='minimizedAffinity'):
    """{p<NNN>_m<MMMM>: value} from eval_out/<tag>/<engine_dir>/pocket*_<mode>.sdf.

    read_dock_tags already unions every docking pass (_rest, _lost) and applies docked_names.txt.
    Penalties are NOT filtered here -- the caller drops them once, over all six values at a time,
    so a molecule excluded for being out-of-the-box can be counted rather than vanishing silently.
    """
    raw = read_dock_tags(os.path.join(EVAL_OUT, tag), engine_dir, mode=mode)
    out = {}
    for k, v in raw.items():
        f = _raw(v.get(sd_tag))
        if f is not None:
            out[k] = f
    return out


def model_records(entry):
    """-> (records, provenance) for one generative model. records key on p<NNN>_m<MMMM>."""
    tag = (entry.get('ids') or {}).get('eval_out')
    sbdd_pt = model_registry.source_for(entry, 'sbdd')
    pose_pt = model_registry.source_for(entry, 'pose') or sbdd_pt

    # Key space is the .pt the SDFs were exported from; metrics come from the sbdd .pt so the Vina
    # columns reproduce the existing F1 row exactly.
    keyed = keyed_entries(pose_pt)
    remapped = os.path.normpath(pose_pt) != os.path.normpath(sbdd_pt)
    if remapped:
        keyed = remap_by_pose(keyed, sbdd_pt)

    eng = {}
    for name, subdir in ENGINES:
        for mode in ('score', 'dock'):
            eng[(name, mode)] = engine_tags(tag, subdir, mode)
    cnn = {ck: engine_tags(tag, sub, mode, sd_tag) for ck, sub, mode, sd_tag, _ in CNN_COLS}

    records, n_vina, n_pen = {}, 0, 0
    for key, e in keyed.items():
        vs, _vm, vd = get_vina(e.get('vina'))
        vs, vd = _raw(vs), _raw(vd)
        if vs is None or vd is None:
            continue
        n_vina += 1
        vals = {(n, m): eng[(n, m)].get(key) for n, _ in ENGINES for m in ('score', 'dock')}
        cvals = {ck: cnn[ck].get(key) for ck, *_ in CNN_COLS}
        if any(v is None for v in vals.values()) or any(v is None for v in cvals.values()):
            continue
        # Penalty screening covers the kcal/mol affinities ONLY -- see AFFINITY_KEYS.
        if any(abs(v) >= PENALTY for v in [vs, vd] + list(vals.values())):
            n_pen += 1                       # out-of-the-box grid penalty, not an affinity
            continue
        rec = {'pk': int(key[1:4]), 'vina_score': vs, 'vina_dock': vd}
        for name, _ in ENGINES:
            rec[f'{name}_score'] = vals[(name, 'score')]
            rec[f'{name}_dock'] = vals[(name, 'dock')]
        rec.update(cvals)
        chem = e.get('chem_results') or {}
        rec['qed'] = _f(chem.get('qed'))
        rec['sa'] = _f(chem.get('sa'))
        rec['fp'] = fingerprint(e.get('mol'))
        records[key] = rec

    prov = {'tag': tag, 'sbdd_pt': sbdd_pt, 'pose_pt': pose_pt, 'remapped': remapped,
            'n_keys': len(keyed), 'n_vina': n_vina, 'n_pen': n_pen, 'n_final': len(records)}
    return records, prov


def fingerprint(mol):
    if mol is None:
        return None
    try:
        return Chem.RDKFingerprint(mol)              # RDKit default fpSize = 2048, as in F1
    except Exception:
        return None


def reference_records(entry):
    """The Reference row. eval_out/native/ has no engine output at all, so this is a second path.

    Vina still comes from the reference .pt (same as every other row). Vinardo and gnina come from
    results/reference_protocol/, which ran the IDENTICAL commands -- see REF_SCORE_DIR above -- and
    is keyed by pocket index rather than by p<NNN>_m<MMMM>, since there is one molecule per pocket.
    """
    score, cnn_score = {}, {}
    for f in sorted(glob.glob(f'{REF_SCORE_DIR}/pocket*.csv')):
        for r in csv.DictReader(open(f)):
            pk = int(r['pocket_idx'])
            score[pk] = {'vinardo': _f(r.get('vinardo_score_only')),
                         'gnina': _f(r.get('gnina_score_only'))}
            # bounded pK, so _raw not _f: it must never be screened as a grid penalty
            cnn_score[pk] = _raw(r.get('gnina_CNNaffinity'))

    def dock_tags(suffix, sd_tag='minimizedAffinity', bounded=False):
        out = {}
        for f in sorted(glob.glob(f'{REF_DOCK_DIR}/poses/pocket*_{suffix}.sdf')):
            if not os.path.getsize(f):
                continue
            pk = int(os.path.basename(f)[len('pocket'):].split('_')[0])
            m = next(iter(Chem.SDMolSupplier(f, sanitize=False, removeHs=False)), None)
            if m is not None and m.HasProp(sd_tag):
                out[pk] = _raw(m.GetProp(sd_tag)) if bounded else _f(m.GetProp(sd_tag))
        return out

    dock = {'vinardo': dock_tags('vinardo'), 'gnina': dock_tags('gnina')}
    cnn = {'cnnaff_score': cnn_score,
           'cnnaff_dock': dock_tags('gnina', 'CNNaffinity', bounded=True)}

    records, n_vina, n_pen = {}, 0, 0
    for pi, pocket in enumerate(load_grouped(model_registry.source_for(entry, 'sbdd'))):
        for mi, e in enumerate(pocket):
            vs, _vm, vd = get_vina(e.get('vina'))
            vs, vd = _raw(vs), _raw(vd)
            if vs is None or vd is None:
                continue
            n_vina += 1
            vals = {(n, 'score'): (score.get(pi) or {}).get(n) for n, _ in ENGINES}
            vals.update({(n, 'dock'): dock[n].get(pi) for n, _ in ENGINES})
            cvals = {ck: cnn[ck].get(pi) for ck, *_ in CNN_COLS}
            if any(v is None for v in vals.values()) or any(v is None for v in cvals.values()):
                continue
            if any(abs(v) >= PENALTY for v in [vs, vd] + list(vals.values())):
                n_pen += 1
                continue
            rec = {'pk': pi, 'vina_score': vs, 'vina_dock': vd}
            for name, _ in ENGINES:
                rec[f'{name}_score'] = vals[(name, 'score')]
                rec[f'{name}_dock'] = vals[(name, 'dock')]
            rec.update(cvals)
            chem = e.get('chem_results') or {}
            rec['qed'] = _f(chem.get('qed'))
            rec['sa'] = _f(chem.get('sa'))
            rec['fp'] = fingerprint(e.get('mol'))
            records[f'p{pi:03d}_m{mi:04d}'] = rec

    prov = {'tag': 'native (reference_protocol)', 'sbdd_pt': model_registry.source_for(entry, 'sbdd'),
            'pose_pt': REF_DOCK_DIR, 'remapped': False,
            'n_keys': n_vina, 'n_vina': n_vina, 'n_pen': n_pen, 'n_final': len(records)}
    return records, prov


def aggregate_row(records, ref_dock, is_reference):
    """-> {group_key: (avg, med)} plus n_pockets / n_mols.

    TWO ESTIMATORS live in one row, deliberately, and the txt header says so:
      * the affinity, gap, QED and SA columns pool MOLECULES across pockets;
      * HA and Div are computed per POCKET and then averaged across pockets, because neither is
        defined on a single molecule -- HA is a rate and Div needs a pair.
    """
    mols = list(records.values())
    for r in mols:
        for name in ('vina',) + tuple(n for n, _ in ENGINES) + ('cnnaff',):
            r[f'{name}_gap'] = abs(r[f'{name}_score'] - r[f'{name}_dock'])

    row = {k: agg([r.get(k) for r in mols]) for k in POOLED}

    by_pocket = {}
    for r in mols:
        by_pocket.setdefault(r['pk'], []).append(r)

    if is_reference:
        # One molecule per pocket: HA would be "does the reference beat itself" and Div needs a
        # pair. Both are undefined, not zero -- same treatment as the F1 table's reference row.
        row['ha'] = (None, None)
        row['div'] = (None, None)
        n_ha = n_div = 0
    else:
        ha = [x for x in (pocket_ha(v, ref_dock.get(i)) for i, v in by_pocket.items())
              if x is not None]
        dv = [x for x in (pocket_div(v) for v in by_pocket.values()) if x is not None]
        row['ha'] = agg(ha)
        row['div'] = agg(dv)
        n_ha, n_div = len(ha), len(dv)

    extra = {'signed': {}, 'n_pk_ha': n_ha, 'n_pk_div': n_div}
    for name in ('vina',) + tuple(n for n, _ in ENGINES) + ('cnnaff',):
        d = [r[f'{name}_score'] - r[f'{name}_dock'] for r in mols]
        extra['signed'][name] = float(np.median(d)) if d else None
    # The displayed |dG|cnn is in pK. The kcal/mol gnina gap it replaced is kept here so the
    # CSV still carries a gap on the same scale as |dV| and |dS|.
    extra['gnina_gap_aff'] = agg([r['gnina_gap'] for r in mols])
    # Largest magnitude that survived the penalty filter. 1e3 is the repo's out-of-the-box
    # threshold, not a claim that everything under it is a real affinity -- a +600 clash is still
    # not one, and this column is how the reader sees that rather than trusting the filter.
    vals = [abs(r[k]) for r in mols for k in AFFINITY_KEYS]
    extra['max_abs'] = max(vals) if vals else None
    return row, len(by_pocket), len(mols), extra


# ----------------------------------------------------------------------------- output

LW = 22                                  # label column width
CW = 9                                   # one numeric cell
W = LW + 6 + 7 + 2 * CW * len(GROUPS)    # banner width == the widest content line


def render(rows):
    """Two-level header: group name centred over its avg/med pair (build_gnina_comparison idiom)."""
    lines = [f'{"Model":<{LW}}{"n_pk":>6}{"n_mol":>7}'
             + ''.join(f'{lab:^{2 * CW}}' for _, lab, _ in GROUPS),
             f'{"":<{LW}}{"":>6}{"":>7}'
             + ''.join(f'{"avg":>{CW}}{"med":>{CW}}' for _ in GROUPS),
             '-' * W]
    for label, row, n_pk, n_mol in rows:
        line = f'{label:<{LW}}{n_pk:>6}{n_mol:>7}'
        for k, _, nd in GROUPS:
            a, m = row[k]
            line += ''.join(f'{fmt(v, nd):>{CW}}' for v in (a, m))
        lines.append(line.rstrip())
    return lines


# `rank` is carried explicitly because row order here is DATA-DERIVED (sorted by HA), unlike every
# other table in results/comparison/, whose order comes from the registry precisely so that a diff
# of two versions cannot confuse a re-ordering with a result change. With `rank` in the file, that
# distinction is recoverable from the CSV even after a re-dock reshuffles the rows.
CSV_HEADER = (['rank', 'id', 'label', 'n_pockets', 'n_mols']
              + [f'{k}_{s}' for k, _, _ in GROUPS for s in ('avg', 'med')]
              + ['n_pockets_ha', 'n_pockets_div',
                 'gnina_gap_aff_avg', 'gnina_gap_aff_med',
                 'vina_gap_signed_med', 'vinardo_gap_signed_med', 'gnina_gap_signed_med',
                 'cnnaff_gap_signed_med', 'max_abs_affinity'])


def write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        for rank, (mid, label, row, n_pk, n_mol, ex) in enumerate(rows, 1):
            vals = []
            for k, _, nd in GROUPS:
                a, m = row[k]
                vals += [fmt(a, nd), fmt(m, nd)]
            sg = ex['signed']
            ga, gm = ex['gnina_gap_aff']
            vals += [ex['n_pk_ha'], ex['n_pk_div'], fmt(ga, 3), fmt(gm, 3),
                     fmt(sg.get('vina'), 3), fmt(sg.get('vinardo'), 3), fmt(sg.get('gnina'), 3),
                     fmt(sg.get('cnnaff'), 3), fmt(ex.get('max_abs'), 1)]
            w.writerow([rank, mid, label, n_pk, n_mol] + vals)


def header_lines(provs):
    bar = '=' * W
    sub = '-' * W
    L = [bar, 'MAIN TABLE - BINDING AFFINITY ACROSS THREE SCORING FUNCTIONS + MOLECULAR PROPERTIES',
         bar, '',
         'Affinities are kcal/mol, LOWER is better. CNNaff / HA / QED / SA / Div: HIGHER is better.',
         '',
         '!! THE THREE AFFINITY BLOCKS ARE NOT THREE INDEPENDENT MEASUREMENTS.',
         '   gnina\'s affinity IS the Vina 1.1.2 scoring function -- the CNN does not enter that',
         '   number -- so the gnina and Vina columns are two implementations of one functional form.',
         '   Measured on these molecules, gnina S vs Vina S is Spearman +0.94..+0.96 with a median',
         '   |diff| of 0.44-0.52 kcal/mol, and gnina S vs Vinardo S is r +0.95..+0.97. Ranking a',
         '   model by three of these columns is close to ranking it three times by the same thing.',
         '   CNNaff is the exception: a 3-D CNN regression on experimental binding constants, not a',
         '   closed-form Vina-family expression, and it agrees with the others far less',
         '   (|rho| 0.57..0.81). It is in the table for exactly that reason.',
         '',
         'CNNaff S / CNNaff D  gnina CNNaffinity for the same two poses as gnina S / gnina D.',
         '   Units are pK (pKd/pKi-like), so HIGHER is better -- the opposite direction to every',
         '   affinity column beside it. It is NOT a unit conversion of the gnina affinity: a',
         '   conversion would force gnina_aff == -1.364 * CNNaff (dG = -2.303*R*T*pK at 298 K)',
         '   exactly, and the measured residual sd is 5.7-6.5 kcal/mol with no linear map doing',
         '   better. There is deliberately no |d| column for it, because a gap in pK units is not',
         '   commensurable with the kcal/mol gaps beside it.',
         '',
         'Score  the molecule\'s OWN generated pose, rescored in place (--score_only, no box).',
         'Dock   a full re-search of the pocket.',
         '|dV| |dS|   per-molecule abs(Score - Dock), then averaged across molecules. kcal/mol.',
         '       NOT avg(Score) - avg(Dock): on this dataset the paired difference and the',
         '       difference of aggregates carry OPPOSITE SIGNS (scripts/analyze_score_vs_dock.py).',
         '|dG|cnn     the same construction on CNNaff, so it is in pK UNITS, not kcal/mol.',
         '       DO NOT read it against |dV| or |dS| -- the three are not on one scale. Using CNNaff',
         '       for the gnina gap does remove a real mismatch: the kcal/mol version differenced a',
         '       CNN-SELECTED pose against a Vina-family score, which is why it showed dock as much',
         '       WORSE than score for AliDiff and IPDiff. Both sides of |dG|cnn come from the CNN.',
         '       But that fix has a price -- see the circularity warning below. The kcal/mol gnina',
         '       gap this column replaced is preserved in main_table.csv as gnina_gap_aff_avg/med.',
         'HA     % of a pocket\'s molecules whose Vina Dock beats that pocket\'s REFERENCE ligand.',
         'Div    mean pairwise Tanimoto distance (1 - similarity, 2048-bit RDKit fp) within a pocket.',
         '',
         'POPULATION. A molecule is counted only if ALL SIX affinities exist -- Vina, Vinardo and',
         'gnina, each in both Score and Dock -- plus both CNNaff readings. Every column including',
         'QED / SA / Div / HA is computed on that one intersected set, so no column carries a',
         f'denominator of its own. Values with |E| >= {PENALTY:.0f} kcal/mol are out-of-the-box grid',
         'penalties, not affinities, and are treated as not-measured; that screen applies to the six',
         'kcal/mol columns ONLY, never to CNNaff, which is a bounded pK and cannot express one.',
         'Section 0 below reports what each stage dropped.',
         '',
         'TWO ESTIMATORS, on purpose. The affinity, gap, QED and SA columns pool MOLECULES across',
         'pockets. HA and Div are computed per POCKET and then averaged over pockets, because',
         'neither is defined on one molecule -- HA is a rate, Div needs a pair. Do not read an',
         'HA/Div (avg) against an affinity (avg) as if they weighted the data the same way.',
         '',
         '!! THE VINA COLUMNS COME FROM A DIFFERENT PROTOCOL THAN THE OTHER TWO.',
         '   Vina            : each model\'s own sampling .pt -- box = the GENERATED ligand\'s own',
         '                     bounding box + 5 A buffer, exhaustiveness 16.',
         '   Vinardo / gnina : eval_out/ -- box = --autobox_ligand <CRYSTAL ligand>, exhaustiveness 8.',
         '   A different box and search depth find different minima. Compare models WITHIN a column;',
         '   never read across the three as if they were one experiment on one protocol.',
         '',
         '!! VINARDO IS NOT ON THE VINA SCALE. Both print "kcal/mol". Vinardo drops Vina\'s gauss2',
         '   term, refits the steric/hydrophobic/h-bond functions, and weights the rotatable-bond',
         '   penalty at 0 where Vina uses 1.923, so it does not charge flexible ligands the entropy',
         '   penalty Vina does. RANK MODELS WITHIN THE VINARDO COLUMNS. Never difference a Vinardo',
         '   number against a Vina or gnina one.',
         '',
         '!! |dG|cnn IS CIRCULAR AND WILL LOOK SMALL FOR EVERY MODEL. gnina runs --num_modes 1 with',
         '   CNN rescoring, so the surviving dock pose is the one the CNN ranked best; scoring that',
         '   pose with the CNN again is close to asking the selector to grade its own choice. A near',
         '   zero |dG|cnn therefore means "the CNN kept a pose it already liked", NOT "re-docking',
         '   could not improve this molecule" -- which is what a small |dV| or |dS| would mean.',
         '   CNNaff D in particular must never be read as "binding strength after re-docking".',
         '   Read the gap columns within-engine only, and treat this one as the weakest of the three.',
         '',
         'The gap columns are ABSOLUTE, so they fold together "docking improved this pose" and',
         '"docking could not reach this distorted pose" -- the sign correlates with strain',
         '(analyze_score_vs_dock.py). main_table.csv carries the signed median of (Score - Dock)',
         'per engine for anyone who needs the direction back.',
         '',
         'ROW ORDER: Reference pinned top, our own arms pinned bottom (Vina Only then Vina+Guide),',
         'and the BASELINE BLOCK between them sorted ASCENDING by HA (avg), so binding strength',
         'increases down that block. Our rows are pinned rather than sorted, so their position',
         'carries NO ranking information -- read their HA against the baseline block above, not',
         'against their own placement. NOTE the sort makes baseline order A FUNCTION OF THE NUMBERS,',
         'unlike every other table in results/comparison/, which orders rows from the registry so',
         'that a re-dock can never silently reshuffle them. Diff two versions of this file and a',
         're-ordering looks the same as a result change -- main_table.csv therefore carries an',
         'explicit `rank` column so the distinction stays recoverable.',
         '']

    L += [sub, '0. COVERAGE AND PROVENANCE', sub, '',
          'n_keys   molecules with a 3-D pose in the .pt the SDFs were exported from',
          'n_vina   of those, how many carry both a Vina Score and a Vina Dock',
          'n_pen    of those, how many were dropped for holding a grid penalty in ANY of the six',
          'n_mol    the table\'s population: measured by all three engines in both states',
          'max|E|   the largest affinity magnitude that SURVIVED the filter. A value in the',
          '         hundreds is still a clash, not a binding energy -- the filter removes',
          '         out-of-the-box artefacts, it does not certify what remains',
          'remap    the SDF key space and the metric .pt are different files, joined on the',
          '         generated pose\'s coordinates (see remap_by_pose)',
          '',
          'Rows are in table order, not registry order, so this section lines up with section 1.',
          '']
    L.append(f'{"Model":<{LW}}{"n_keys":>8}{"n_vina":>8}{"n_pen":>7}{"n_mol":>8}{"max|E|":>9}  '
             f'{"remap":<6}  eval_out / source')
    L.append('-' * W)
    for label, p in provs:
        L.append(f'{label:<{LW}}{p["n_keys"]:>8}{p["n_vina"]:>8}{p["n_pen"]:>7}{p["n_final"]:>8}'
                 f'{fmt(p.get("max_abs"), 1):>9}  '
                 f'{"yes" if p["remapped"] else "":<6}  {p["tag"]}  <-  {p["sbdd_pt"]}')
    L += ['',
          'WHY THIS TABLE\'S Vina D (avg) CAN DIFFER FROM comparison_overall.csv. The F1 table applies',
          'no penalty filter, so a molecule whose Vina Dock is a grid penalty enters its MEAN. There',
          'is exactly one such molecule in this roster -- Ours (Vina+Guide) p061_m0060, Vina Dock',
          '+1559.257 kcal/mol -- and on its own it moves that row\'s F1 Vina D (avg) from -10.009 to',
          '-9.821. This table drops it, so read -10.010 here against -9.821 there as the same',
          'quantity measured with and without one out-of-the-box artefact. Every MEDIAN agrees',
          'exactly across the two tables, which is what a single outlier predicts.',
          '']
    return L


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out_dir', default='results/comparison/f1_sbdd/main_table')
    p.add_argument('--registry', default='configs/models.json')
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    entries = [model_registry.by_id(i, a.registry) for i in ROW_IDS]

    built = []
    ref_dock = {}
    for e in entries:
        is_ref = e['tier'] == 'reference'
        label = model_registry.label_for(e, 'sbdd')
        recs, prov = reference_records(e) if is_ref else model_records(e)
        if is_ref:
            # The HA yardstick: this pocket's reference-ligand Vina Dock.
            ref_dock = {r['pk']: r['vina_dock'] for r in recs.values()}
        built.append((e, label, recs, is_ref, prov))
        print(f'  {label:24s} n_pk={len({r["pk"] for r in recs.values()}):3d} '
              f'n_mol={len(recs):5d}{"  (pose remap)" if prov["remapped"] else ""}')

    rows = []
    for e, label, recs, is_ref, prov in built:
        row, n_pk, n_mol, extra = aggregate_row(recs, ref_dock, is_ref)
        prov['max_abs'] = extra.get('max_abs')
        rows.append({'id': e['id'], 'label': label, 'row': row, 'n_pk': n_pk,
                     'n_mol': n_mol, 'is_ref': is_ref, 'extra': extra, 'prov': prov})

    # Reference pinned top, our own arms pinned bottom in OURS_LAST order, baselines in between
    # sorted weakest-binding first. A baseline whose HA never resolved sorts to the end of that
    # block rather than silently landing at the top of a "weakest first" table.
    by_id = {r['id']: r for r in rows}
    ref = [r for r in rows if r['is_ref']]
    ours = [by_id[i] for i in OURS_LAST if i in by_id]
    pinned = {r['id'] for r in ref} | set(OURS_LAST)
    mid = sorted([r for r in rows if r['id'] not in pinned],
                 key=lambda r: (r['row']['ha'][0] is None, r['row']['ha'][0] or 0.0))
    rows = ref + mid + ours

    L = header_lines([(r['label'], r['prov']) for r in rows])
    L += ['-' * W, '1. MAIN TABLE', '-' * W, '']
    L += render([(r['label'], r['row'], r['n_pk'], r['n_mol']) for r in rows])
    L += ['',
          f'Reference (native) is 1 molecule per pocket, so its per-molecule and per-pocket',
          f'statistics coincide; its HA (it cannot beat itself) and Div (needs a pair) are blank,',
          f'not zero. Its Vinardo/gnina numbers come from {REF_DOCK_DIR} and',
          f'{REF_SCORE_DIR}, which ran the identical commands.',
          '',
          'Source:     scripts/build_main_table.py',
          'Regenerate: python scripts/build_main_table.py',
          '']

    txt = os.path.join(a.out_dir, 'main_table.txt')
    csvp = os.path.join(a.out_dir, 'main_table.csv')
    with open(txt, 'w') as f:
        f.write('\n'.join(L) + '\n')
    write_csv(csvp, [(r['id'], r['label'], r['row'], r['n_pk'], r['n_mol'], r['extra'])
                     for r in rows])
    print(f'\nWrote:\n  {txt}\n  {csvp}')


if __name__ == '__main__':
    main()
