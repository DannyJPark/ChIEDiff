#!/usr/bin/env python3
"""Molecular property panel for the F1-F4 roster: QED / SA / LogP / TPSA / MW / Lipinski / Diversity.

WHY THIS EXISTS. results/comparison/f1_sbdd/comparison_overall.csv already reports QED, SA, HA%
and Diversity. It does not report LogP, TPSA, molecular weight or Lipinski -- the other four of
the six properties that the SBDD literature reports as a panel. The reference for the panel is
results/papers/SGEDiff_*.pdf, whose Fig. 3 (p.9) and "Physicochemical properties analysis" (p.11)
cover exactly "QED, SA, LogP, Lipinski, TPSA, and MW" -- but ONLY as distribution plots. That
paper ships no numeric property table at all, and never reports Diversity. This builder is the
numeric table it lacks, plus a Fig. 3-equivalent figure so the two can be read side by side.

MEASURED, NOT COPIED. 16 of 17 source .pt files already carry chem_results = {qed, sa, logp,
lipinski, ring_size}; MolCRAFT carries only {qed, sa}; NONE carries TPSA or MW. Rather than mix
four papers' RDKit versions with two freshly computed columns, every value here is recomputed
from the stored RDKit `mol` under ONE interpreter -- which also fills MolCRAFT's LogP/Lipinski
instead of leaving a hole. Gate G4 below then checks the recomputation against what each file
stored, so "recomputed" is a verified claim rather than an assumption.

POPULATION. Docking-successful molecules only -- the repo-wide rule (results/comparison/README.md,
"no exceptions"), which is also what puts this table's QED/SA on the same denominator as the
published F1 row and so makes gates G1-G3 meaningful. For OUR rows that is a survivor population:
the 15-28% fragmented share never docks and is absent here, exactly as F1's HA% caveat describes
(analysis/RESULTS_pose_fidelity_2026-07.md). Stated, not hidden.

CONVENTIONS (all pinned, all printed into the txt header):
    QED       rdkit.Chem.QED.qed                                        higher better
    SA        sascorer.compute_sa_score -> NORMALISED round((10-sa)/9,2) higher better
    LogP      Crippen.MolLogP (scoring_func.get_logp)                    window -2..5
    TPSA      Descriptors.TPSA, A^2                                      window 50..150
    MW        Descriptors.MolWt (average mass). ExactMolWt also dumped per molecule.
    Lipinski  scoring_func.obey_lipinski -> COUNT of rules passed, 0-5.  higher better
              Reported as mean count AND % of molecules passing all five.
    Div       per pocket, mean pairwise Tanimoto DISTANCE over 2048-bit Chem.RDKFingerprint,
              then avg/med across pockets. Undefined for the 1-molecule-per-pocket reference.

Normalised SA and Lipinski-as-0..5-count are the TargetDiff/KGDiff conventions SGEDiff inherits
(it states neither, but its p.11 "all models yielded median values between 0.5 and 0.6" with
higher-is-better is only consistent with the normalised form). Diversity is this repo's own F1
definition, reused verbatim via build_comparison_tables.pocket_div.

ENVIRONMENT. Run under `kgdiff` (RDKit 2024.03.6). NOT `posecheck`: CLAUDE.md pins that env's
RDKit 2026.3.4 and requires it not leak into kgdiff. The RDKit version is stamped into every
output file.

Usage:
    conda run -n kgdiff python scripts/build_property_tables.py --jobs 8
    conda run -n kgdiff python scripts/build_property_tables.py --ids reference,targetdiff,ours_vina
"""

import argparse
import csv
import gzip
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)     # build_comparison_tables, model_registry
sys.path.insert(0, _ROOT)     # utils.evaluation

from rdkit import Chem, RDLogger                                         # noqa: E402
from rdkit.Chem import QED, Crippen, Descriptors                         # noqa: E402
import rdkit                                                             # noqa: E402

RDLogger.DisableLog('rdApp.*')

import build_comparison_tables as bct                                    # noqa: E402
import model_registry                                                    # noqa: E402
from utils.evaluation.sascorer import compute_sa_score                   # noqa: E402
from utils.evaluation.scoring_func import obey_lipinski, get_logp        # noqa: E402


# ------------------------------------------------------------------ metric definitions
PROPS = ['qed', 'sa', 'logp', 'tpsa', 'mw', 'lipinski']
PROP_LABEL = {'qed': 'QED', 'sa': 'SA', 'logp': 'LogP', 'tpsa': 'TPSA',
              'mw': 'MW', 'lipinski': 'Lip'}
PROP_ND = {'qed': 3, 'sa': 3, 'logp': 2, 'tpsa': 1, 'mw': 1, 'lipinski': 2}

# The 9-row roster the manuscript prints (paper/make_tables.sh ROSTER), with its relabel.
PAPER_ROSTER = ['reference', 'pocket2mol', 'targetdiff', 'ipdiff', 'alidiff',
                'molcraft', 'pidiff_retrain', 'kgdiff', 'ours_vina']
from gated_energy_diffusion import MODEL_NAME
PAPER_RELABEL = {'reference': 'Test set', 'ours_vina': MODEL_NAME}

# Tier 'ablation' but a reported F1/F2/F3 row -- for_family's tier filter would drop it.
EXTRA_IDS = ['novdw']

F1_CSV = 'results/comparison/f1_sbdd/comparison_overall.csv'


def build_roster(registry, ids=None):
    """Every model that is a row in a finished F1-F4 table, in registry `order`.

    Derived from configs/models.json rather than hard-coded: the union over the four families of
    the reportable reference/core/extended entries, plus EXTRA_IDS. `bind` enters through nci
    (its sbdd status is 'deferred', which is a reporting decision -- its .pt is properly grouped,
    100 pockets x ~10 molecules).
    """
    if ids:
        return [model_registry.by_id(i, registry) for i in ids]
    seen, out = set(), []
    for fam in ('sbdd', 'pose', 'nci', 'delta'):
        for m in model_registry.for_family(fam, tiers=('reference', 'core', 'extended'),
                                           statuses=('ok',), sections=None, path=registry):
            if m['id'] not in seen:
                seen.add(m['id'])
                out.append(m)
    for i in EXTRA_IDS:
        if i not in seen:
            seen.add(i)
            out.append(model_registry.by_id(i, registry))
    return sorted(out, key=lambda m: (m.get('order', 10 ** 6), m['id']))


# ------------------------------------------------------------------ per-molecule extraction
def sanitized_copy(rdmol):
    """A sanitized copy of the molecule, or None when it is not valid chemistry.

    A molecule that fails RDKit sanitization is not a measurable subject, and the failure is NOT
    uniform across the panel: on AliDiff's four neutral 4-valent-nitrogen molecules QED, SA and
    Lipinski raise outright, while Crippen.MolLogP quietly returns a number typed from an
    impossible valence state -- up to 3.7 log units away from what the file itself stored. Taking
    the number for one property and a blank for the next is the worst of both, so a molecule that
    cannot be sanitized contributes to NO property and is counted instead.
    """
    try:
        m = Chem.Mol(rdmol)
        Chem.SanitizeMol(m)
        return m
    except Exception:
        return None


def extract_props(mol, need_fp=False):
    """Extractor handed to build_comparison_tables.normalize_model.

    Starts from extract_mol (vina_score/min/dock, the STORED qed/sa, and the Div fingerprint) so
    the pocket alignment and the docked-only filter behave identically, then recomputes all six
    properties from the RDKit mol. The stored values are kept under stored_* purely to feed gate
    G4. Each property is guarded separately: one pathological molecule must not null the other
    five for that row.

    Only docking-successful molecules are measured -- everything downstream is docked_only, and
    for our rows that skips ~20% of the entries.
    """
    md = bct.extract_mol(mol, need_fp)
    chem = mol.get('chem_results') or {}
    md['stored_qed'] = md['qed']
    md['stored_sa'] = md['sa']
    md['stored_logp'] = bct._f(chem.get('logp'))
    md['stored_lipinski'] = bct._f(chem.get('lipinski'))

    for k in PROPS + ['exact_mw', 'heavy']:
        md[k] = None
    md['calc_ok'] = None
    md['unsanitizable'] = False

    rdmol = mol.get('mol')
    if md['vina_dock'] is None or rdmol is None:
        return md

    sane = sanitized_copy(rdmol)
    if sane is None:
        md['calc_ok'] = False
        md['unsanitizable'] = True
        return md

    ok = True
    for key, fn in (('qed', lambda m: float(QED.qed(m))),
                    ('sa', lambda m: float(compute_sa_score(m))),
                    ('logp', lambda m: float(get_logp(m))),
                    ('tpsa', lambda m: float(Descriptors.TPSA(m))),
                    ('mw', lambda m: float(Descriptors.MolWt(m))),
                    ('exact_mw', lambda m: float(Descriptors.ExactMolWt(m))),
                    ('lipinski', lambda m: float(obey_lipinski(m))),
                    ('heavy', lambda m: float(m.GetNumHeavyAtoms()))):
        try:
            md[key] = fn(rdmol)
        except Exception:
            # Retry on the sanitized copy, never instead of the raw molecule. AR's stored mols
            # carry no ring perception, so Descriptors.TPSA raises "RingInfo not initialized" on
            # 1066 of them while every other descriptor is fine; the copy has ring info and
            # returns the right number. Values that already worked keep their raw-mol result, so
            # the agreement with each file's own chem_results stays exact (gate G4).
            try:
                md[key] = fn(sane)
            except Exception:
                ok = False
    md['calc_ok'] = ok
    return md


# ------------------------------------------------------------------ aggregation
def _clean(values):
    return [float(v) for v in values
            if v is not None and not (isinstance(v, float) and np.isnan(v))]


def stats(values):
    """avg / std / med / q25 / q75 over the non-missing values. std is the population sd."""
    v = _clean(values)
    if not v:
        return {k: None for k in ('avg', 'std', 'med', 'q25', 'q75', 'n')}
    a = np.asarray(v, dtype=float)
    return {'avg': float(a.mean()), 'std': float(a.std()), 'med': float(np.median(a)),
            'q25': float(np.percentile(a, 25)), 'q75': float(np.percentile(a, 75)),
            'n': int(a.size)}


def _maxabs(pairs):
    """max |new - stored| over the molecules where the file stored a value; None if none did."""
    d = [abs(n - s) for n, s in pairs if n is not None and s is not None]
    return max(d) if d else None


def measure_one(job):
    """Load one model's .pt and reduce it to plain picklable data.

    Diversity is computed HERE because it needs the RDKit fingerprint objects, which must not
    cross a process boundary. Everything returned is float / int / str.
    """
    entry, src, full2idx, dir2idx, idx2name = job
    mid, label = entry['id'], entry['label']
    t0 = time.time()

    obj = bct.load_pt(src)
    per_pocket, flat = bct.normalize_model(obj, full2idx, dir2idx,
                                           need_fp=True, extract=extract_props)
    del obj

    valid = bct.docked_only(flat)
    pooled = {p: stats([md[p] for md in valid]) for p in PROPS}
    pooled['heavy'] = stats([md['heavy'] for md in valid])
    pooled['exact_mw'] = stats([md['exact_mw'] for md in valid])

    lip = _clean([md['lipinski'] for md in valid])
    pass5 = 100.0 * sum(1 for x in lip if x >= 5) / len(lip) if lip else None
    pass4 = 100.0 * sum(1 for x in lip if x >= 4) / len(lip) if lip else None

    # Diversity: per pocket first, then avg/med across pockets. Reuses F1's pocket_div verbatim,
    # which is what makes gate G2 a real check on the pocket alignment.
    pocket_div = {i: bct.pocket_div(mols) for i, mols in per_pocket.items()}
    div_vals = [v for v in pocket_div.values() if v is not None]
    div_avg, div_med = bct.agg(div_vals)

    per_target, per_mol = [], []
    for i in sorted(per_pocket):
        pk_valid = bct.docked_only(per_pocket[i])
        if not pk_valid:
            continue
        row = {'pocket_idx': i, 'pocket_name': idx2name.get(i, ''), 'n_mol': len(pk_valid),
               'div': pocket_div.get(i)}
        for p in PROPS:
            s = stats([md[p] for md in pk_valid])
            row[f'{p}_avg'], row[f'{p}_med'] = s['avg'], s['med']
        per_target.append(row)
        # mol_idx is the within-pocket appearance order in THIS .pt (see the dump header note).
        for j, md in enumerate(per_pocket[i]):
            if md['vina_dock'] is None:
                continue
            per_mol.append([i, idx2name.get(i, ''), j, md['heavy'], md['qed'], md['sa'],
                            md['logp'], md['tpsa'], md['mw'], md['exact_mw'], md['lipinski'],
                            md['vina_dock']])

    gate = {
        'qed_avg': pooled['qed']['avg'], 'sa_avg': pooled['sa']['avg'],
        'div_avg': div_avg, 'div_med': div_med,
        'n_mols': len(valid), 'n_pockets': len(per_pocket),
        'd_qed': _maxabs([(md['qed'], md['stored_qed']) for md in valid]),
        'd_sa': _maxabs([(md['sa'], md['stored_sa']) for md in valid]),
        'd_logp': _maxabs([(md['logp'], md['stored_logp']) for md in valid]),
        'd_lipinski': _maxabs([(md['lipinski'], md['stored_lipinski']) for md in valid]),
        'n_unsanitizable': sum(1 for md in valid if md.get('unsanitizable')),
        'n_calc_fail': sum(1 for md in valid
                           if md['calc_ok'] is False and not md.get('unsanitizable')),
        'n_no_mol': sum(1 for md in valid if md['calc_ok'] is None),
    }

    n_unmeasurable = sum(1 for md in valid if md['qed'] is None)

    return {'id': mid, 'label': label, 'source': src,
            'n_pockets': len(per_pocket), 'n_mols': len(valid),
            'n_unmeasurable': n_unmeasurable,
            'pooled': pooled, 'lipinski_pass5_pct': pass5, 'lipinski_pass4plus_pct': pass4,
            'div_avg': div_avg, 'div_med': div_med, 'n_pockets_div': len(div_vals),
            'per_target': per_target, 'per_mol': per_mol, 'gate': gate,
            'secs': time.time() - t0}


# ------------------------------------------------------------------ validation gates
def read_f1(path):
    """{id: row} from f1_sbdd/comparison_overall.csv, the table these numbers must agree with."""
    if not os.path.isfile(path):
        return {}
    with open(path, newline='', encoding='utf-8') as fh:
        return {r['id']: r for r in csv.DictReader(fh)}


def _cell(row, key):
    v = (row or {}).get(key, '')
    return v.strip() if isinstance(v, str) else v


def run_gates(results, f1, f1_path):
    """G1-G4. Returns (lines, ok). A failure is fatal: a property table that cannot reproduce
    the published QED/SA/Div/n is measuring a different population than it claims to.

    G2 compares at the CSV's own printed precision (3 dp) rather than with a tolerance -- the
    question is whether this builder reproduces the published CELL, and Diversity uses the very
    same fingerprint and the very same pocket_div, so anything but equality means the pocket
    alignment moved.
    """
    L = ['GATE REPORT', '=' * 110, '',
         f'Reference table: {f1_path}',
         '  G1  pooled QED/SA avg reproduce the F1 cell to <= 0.005',
         '  G2  Diversity avg/med reproduce the F1 cell exactly at its printed precision (3 dp)',
         '  G3  n_mols / n_pockets equal the F1 cell exactly (same population, same alignment)',
         '  G4  recomputed QED/LogP/Lipinski equal the stored chem_results where a file stored',
         '      them (QED/LogP <= 1e-6, Lipinski exact). SA is REPORTED not gated: sascorer',
         '      rounds to 2 dp and ring perception drifts across RDKit versions.',
         '']
    hdr = ['Model', 'G1 QED', 'G1 SA', 'G2 Div', 'G3 n', 'G4 dQED', 'G4 dLogP', 'G4 dLip',
           'SA drift', 'unsan', 'calc err']
    rows, ok = [], True
    for r in results:
        g, ref = r['gate'], f1.get(r['id'])
        marks = {}
        if ref is None:
            marks = {k: 'n/a' for k in ('qed', 'sa', 'div', 'n')}
        else:
            for key, gk in (('qed', 'qed_avg'), ('sa', 'sa_avg')):
                c = _cell(ref, gk)
                if c == '' or g[gk] is None:
                    marks[key] = 'n/a'
                else:
                    d = abs(g[gk] - float(c))
                    marks[key] = 'PASS' if d <= 0.005 else f'FAIL {d:.4f}'
            bad = []
            for gk in ('div_avg', 'div_med'):
                c = _cell(ref, gk)
                mine = '' if g[gk] is None else f'{g[gk]:.3f}'
                if c != mine:
                    bad.append(f'{gk} {mine or "-"} vs {c or "-"}')
            marks['div'] = 'PASS' if not bad else 'FAIL ' + '; '.join(bad)
            nbad = [f'{k} {g[k]} vs {_cell(ref, k)}'
                    for k in ('n_mols', 'n_pockets') if str(g[k]) != _cell(ref, k)]
            marks['n'] = 'PASS' if not nbad else 'FAIL ' + '; '.join(nbad)

        def g4(v, tol):
            if v is None:
                return 'not stored'
            return f'{v:.3g}' if v <= tol else f'FAIL {v:.3g}'

        cells = [r['id'], marks['qed'], marks['sa'], marks['div'], marks['n'],
                 g4(g['d_qed'], 1e-6), g4(g['d_logp'], 1e-6), g4(g['d_lipinski'], 0.0),
                 'n/a' if g['d_sa'] is None else f'{g["d_sa"]:.3g}',
                 g['n_unsanitizable'], g['n_calc_fail'] + g['n_no_mol']]
        if any(str(c).startswith('FAIL') for c in cells):
            ok = False
        rows.append(cells)
    L += bct.render_table(hdr, rows, ['l'] + ['l'] * (len(hdr) - 1))
    L += ['', f'GATES: {"ALL PASS" if ok else "FAILED"}', '',
          'unsan     molecules inside the docking-successful set that fail RDKit sanitization.',
          '          They stay in n_mols (the population must match F1) but contribute to NO',
          '          property -- see n_unmeasurable in property_overall.csv. AliDiff has 4, all',
          '          with a neutral 4-valent nitrogen; that is also where its stored LogP and a',
          '          fresh Crippen.MolLogP disagree by up to 3.7, since MolLogP does not raise',
          '          on an impossible valence, it just types it wrong.',
          'calc err  a property still raised after the sanitized-copy retry. Expected: 0.',
          '          AR needed that retry for 1066 molecules whose stored mol carries no ring',
          '          perception, so Descriptors.TPSA alone raised on them.']
    return L, ok


# ------------------------------------------------------------------ writers
def overall_columns():
    cols = ['id', 'label', 'n_pockets', 'n_mols', 'n_unmeasurable']
    for p in PROPS:
        cols += [f'{p}_avg', f'{p}_std', f'{p}_med', f'{p}_q25', f'{p}_q75']
    cols += ['lipinski_pass5_pct', 'lipinski_pass4plus_pct',
             'div_avg', 'div_med', 'n_pockets_div',
             'heavy_avg', 'heavy_med', 'exact_mw_avg', 'source']
    return cols


def _num(x, nd):
    return '' if x is None else f'{x:.{nd}f}'


def write_overall_csv(path, results):
    cols = overall_columns()
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in results:
            row = {'id': r['id'], 'label': r['label'], 'n_pockets': r['n_pockets'],
                   'n_mols': r['n_mols'], 'n_unmeasurable': r['n_unmeasurable'],
                   'source': r['source'],
                   'lipinski_pass5_pct': _num(r['lipinski_pass5_pct'], 1),
                   'lipinski_pass4plus_pct': _num(r['lipinski_pass4plus_pct'], 1),
                   'div_avg': _num(r['div_avg'], 3), 'div_med': _num(r['div_med'], 3),
                   'n_pockets_div': r['n_pockets_div'],
                   'heavy_avg': _num(r['pooled']['heavy']['avg'], 2),
                   'heavy_med': _num(r['pooled']['heavy']['med'], 2),
                   'exact_mw_avg': _num(r['pooled']['exact_mw']['avg'], 2)}
            for p in PROPS:
                s, nd = r['pooled'][p], PROP_ND[p]
                for k in ('avg', 'std', 'med', 'q25', 'q75'):
                    row[f'{p}_{k}'] = _num(s[k], nd)
            w.writerow([row[c] for c in cols])


def write_per_target_csv(path, results):
    cols = ['pocket_idx', 'pocket_name', 'id', 'label', 'n_mol']
    for p in PROPS:
        cols += [f'{p}_avg', f'{p}_med']
    cols += ['div']
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in results:
            for t in r['per_target']:
                row = {'pocket_idx': t['pocket_idx'], 'pocket_name': t['pocket_name'],
                       'id': r['id'], 'label': r['label'], 'n_mol': t['n_mol'],
                       'div': _num(t['div'], 3)}
                for p in PROPS:
                    row[f'{p}_avg'] = _num(t[f'{p}_avg'], PROP_ND[p])
                    row[f'{p}_med'] = _num(t[f'{p}_med'], PROP_ND[p])
                w.writerow([row[c] for c in cols])


PER_MOL_HEADER = [
    '# Per-molecule molecular properties, docking-successful molecules only.',
    '# Built by scripts/build_property_tables.py -- see that file for every convention.',
    '#',
    '# mol_idx is the within-pocket APPEARANCE ORDER in that row\'s own source .pt (the `source`',
    '# column of property_overall.csv, i.e. the registry sources.default). It joins to the',
    '# eval_out/<tag>/sdf/ titles p<pk>_m<idx> ONLY for models whose sources.default is also',
    '# their pose/nci source. ipdiff, molcraft and ours_noguide declare per-family source',
    '# overrides and do NOT join. Check the source column before joining, do not assume.',
    '#',
    '# mw is Descriptors.MolWt (average mass); exact_mw is Descriptors.ExactMolWt.',
    '# sa is the NORMALISED (10-sa)/9 rounded to 2 dp. lipinski is the count of rules passed, 0-5.',
]


def write_per_molecule(path, results, stamp):
    with gzip.open(path, 'wt', newline='', encoding='utf-8') as fh:
        for line in [f'# {stamp}'] + PER_MOL_HEADER:
            fh.write(line + '\n')
        w = csv.writer(fh)
        w.writerow(['id', 'pocket_idx', 'pocket_name', 'mol_idx', 'heavy',
                    'qed', 'sa', 'logp', 'tpsa', 'mw', 'exact_mw', 'lipinski', 'vina_dock'])
        for r in results:
            for (pi, pn, j, heavy, qed, sa, logp, tpsa, mw, emw, lip, vd) in r['per_mol']:
                w.writerow([r['id'], pi, pn, j, _num(heavy, 0), _num(qed, 4), _num(sa, 2),
                            _num(logp, 4), _num(tpsa, 2), _num(mw, 3), _num(emw, 4),
                            _num(lip, 0), _num(vd, 3)])


def _pm(s, nd):
    """avg+-sd in one cell. ASCII only: these .txt files are read on the cluster."""
    if s['avg'] is None:
        return ''
    return f'{s["avg"]:.{nd}f}+-{s["std"]:.{nd}f}'


def _iqr(s, nd):
    if s['med'] is None:
        return ''
    return f'{s["med"]:.{nd}f} [{s["q25"]:.{nd}f},{s["q75"]:.{nd}f}]'


def txt_header(stamp, f1_path):
    return [
        '=' * 130,
        'MOLECULAR PROPERTY PANEL -- QED / SA / LogP / TPSA / MW / Lipinski / Diversity',
        '=' * 130,
        '',
        stamp,
        '',
        'A SUB-TABLE of F1, not a replacement. results/comparison/f1_sbdd/comparison_tables.txt',
        'stays the F1 table of record; these are the SAME molecules under the SAME population',
        'rule, with the four properties F1 does not carry (LogP, TPSA, MW, Lipinski) added and',
        'every value re-measured under one RDKit so no column mixes four papers\' versions.',
        '',
        'POPULATION. Docking-successful molecules only -- the repo-wide rule. External baselines',
        'ship .pt files that already contain only docking-successful molecules, so including our',
        'failures would evaluate us on a wider population. For OUR rows this is therefore a',
        'SURVIVOR population: the 15-28% fragmented share never docks and is absent from these',
        'averages, the same optimism F1 documents for HA%. Quote the fragmentation rate from',
        'analysis/RESULTS_pose_fidelity_2026-07.md alongside any property claim about our rows.',
        '',
        'DEFINITIONS (higher better unless a window is given)',
        '  QED       rdkit.Chem.QED.qed',
        '  SA        utils/evaluation/sascorer.compute_sa_score -> NORMALISED round((10-sa)/9, 2)',
        '  LogP      Crippen.MolLogP                                     window -2..5',
        '  TPSA      Descriptors.TPSA, A^2                               window 50..150',
        '  MW        Descriptors.MolWt (average mass); ExactMolWt is in the per-molecule dump',
        '  Lip       obey_lipinski -> COUNT of the five rules passed, 0-5',
        '  Lip5%     % of molecules passing all five rules',
        '  Div       per pocket, mean pairwise Tanimoto DISTANCE over 2048-bit RDKFingerprint,',
        '            then avg/med across pockets. UNDEFINED for the reference row (1 mol/pocket).',
        '',
        'Normalised SA and Lipinski-as-a-count are the TargetDiff/KGDiff conventions that the',
        'SGEDiff panel (results/papers/SGEDiff_*.pdf, Fig. 3 p.9 and p.11) inherits; that paper',
        'reports these six as DISTRIBUTIONS ONLY and reports no Diversity, so property_distributions',
        'is the figure to read against its Fig. 3 and this table has no counterpart there.',
        '',
        'property_distributions covers the SIX per-molecule properties only, in SGEDiff Fig. 3',
        'order and form (violin + inner box). Diversity is per-POCKET, so it has no place beside',
        'them on one canvas and lives in this table alone.',
        '',
        f'Gates below check this table against {f1_path}.',
        '',
    ]


def txt_table(results, ids=None, relabel=None, title='', note=''):
    keep = [r for r in results if ids is None or r['id'] in ids]
    if ids:
        keep.sort(key=lambda r: ids.index(r['id']))
    relabel = relabel or {}
    L = ['-' * 130, title, '-' * 130, '']
    if note:
        L += [note, '']

    hdr = ['Model', 'n_pk', 'n_mol'] + [PROP_LABEL[p] for p in PROPS] + ['Lip5%', 'Div']
    rows = [[relabel.get(r['id'], r['label']), r['n_pockets'], r['n_mols']]
            + [_pm(r['pooled'][p], PROP_ND[p]) for p in PROPS]
            + [_num(r['lipinski_pass5_pct'], 1), _num(r['div_avg'], 3)]
            for r in keep]
    L += ['AVG +- SD (pooled over docking-successful molecules; Div is per-pocket then averaged)',
          '']
    L += bct.render_table(hdr, rows, ['l'] + ['r'] * (len(hdr) - 1))

    hdr2 = ['Model', 'n_mol'] + [PROP_LABEL[p] for p in PROPS] + ['Div']
    rows2 = [[relabel.get(r['id'], r['label']), r['n_mols']]
             + [_iqr(r['pooled'][p], PROP_ND[p]) for p in PROPS]
             + ['' if r['div_med'] is None else f'{r["div_med"]:.3f}']
             for r in keep]
    L += ['', 'MEDIAN [Q25,Q75]', '']
    L += bct.render_table(hdr2, rows2, ['l'] + ['r'] * (len(hdr2) - 1))
    L += ['']
    return L


def write_txt(path, results, stamp, gate_lines, f1_path):
    L = txt_header(stamp, f1_path)
    L += txt_table(results, title='OVERALL -- every model that is a row in a finished F1-F4 table',
                   note='Row order is the registry `order`, never a sort on the numbers.\n'
                        'bind carries no F1 row (families.sbdd = deferred); it enters here through F3.')
    have = {r['id'] for r in results}
    paper = [i for i in PAPER_ROSTER if i in have]
    if len(paper) == len(PAPER_ROSTER):
        L += txt_table(results, ids=paper, relabel=PAPER_RELABEL,
                       title='PAPER ROSTER -- the nine rows paper/make_tables.sh prints',
                       note='Same numbers, restricted and relabelled to the manuscript roster so a\n'
                            'property claim in the text can be read against one block.')
    L += gate_lines
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')


# ------------------------------------------------------------------ figure
# Built to match SGEDiff Fig. 3 (results/papers/SGEDiff_*.pdf, p. 9): a 2x3 grid of the same six
# properties in the same order, each a pastel violin with a saturated box plot inside it, framed
# axes, no grid, no shaded windows, no legend and no footer. Everything the figure needs a reader
# to know that is not on an axis goes in the caption.
#
# COLOUR CARRIES ROLE, NOT IDENTITY. Identity is the x tick label under every violin, so colour is
# free to mark the only distinction the reader acts on: reference / baseline / ours. SGEDiff gives
# each of its five models its own pastel; at nine models that becomes a rainbow, and the palette
# rule caps a categorical set at eight. These three were validated with the dataviz palette
# checker (light surface, all pairs): CVD dE 16.3, normal-vision dE 18.3, contrast >= 3:1 for all
# three. The neutral fails the chroma floor by construction -- reading as grey is what makes the
# test set look like a reference rather than a competitor.
C_REF, C_BASE, C_OURS = '#565b62', '#2a78d6', '#eb6834'
OURS_IDS = {'ours_vina', 'ours_noguide', 'novdw'}

# SGEDiff Fig. 3's panel order, left to right then top to bottom.
FIG_PANELS = [('qed', 'QED', 'QED'), ('sa', 'SA', 'SA (normalised)'),
              ('logp', 'LogP', 'LogP'), ('lipinski', 'Lipinski', 'Lipinski (rules passed)'),
              ('tpsa', 'TPSA', 'TPSA  ($\\AA^2$)'), ('mw', 'MW', 'MW  (Da)')]

# Untrimmed, MW runs to 1000 Da and LogP past +-20 on a handful of molecules, and every panel
# collapses into a flat band. So the VIOLIN is estimated on the pooled 0.5-99.5 percentile window
# of the models on show -- trimming the KDE input rather than the axis, so each body terminates
# inside the frame instead of being sliced off by it. The BOX PLOT beside it always sees the full
# data, so the median, quartiles and 1.5-IQR whiskers are never affected, and every statistic in
# property_overall.csv is computed on the full data too. Lipinski is exempt: it is a bounded 0-5
# count whose whole range is the point.
FIG_CLIP = (0.5, 99.5)
FIG_NO_CLIP = {'lipinski'}

# Lipinski gets the violin alone. On a 0-5 integer count whose median and Q75 are both 5 for every
# model, the box degenerates into a line pinned to the top of the panel: it hides the violin's
# shoulder at 4 and 3, which is the only thing that separates the models here, and reports nothing
# the violin does not already show.
FIG_NO_BOX = {'lipinski'}


def _mix(hex_color, white):
    """hex_color lightened toward white by `white` in [0,1]."""
    h = hex_color.lstrip('#')
    rgb = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return tuple(c + (1.0 - c) * white for c in rgb)


def make_figure(results, out_png, out_pdf, ids):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    keep = [r for r in results if r['id'] in ids]
    keep.sort(key=lambda r: ids.index(r['id']))
    if not keep:
        return None
    labels = [PAPER_RELABEL.get(r['id'], r['label']) for r in keep]
    bases = [C_REF if r['id'] == 'reference' else (C_OURS if r['id'] in OURS_IDS else C_BASE)
             for r in keep]

    col = {p: i for i, p in enumerate(
        ['heavy', 'qed', 'sa', 'logp', 'tpsa', 'mw', 'exact_mw', 'lipinski'], start=3)}
    series = {p: [[row[col[p]] for row in r['per_mol'] if row[col[p]] is not None]
                  for r in keep] for p in PROPS}

    fig, axes = plt.subplots(2, 3, figsize=(13.6, 8.6), dpi=200)
    x = np.arange(1, len(keep) + 1)

    for ax, (key, title, ylab) in zip(axes.ravel(), FIG_PANELS):
        data = series[key]
        pooled = [v for d in data for v in d]
        if pooled and key not in FIG_NO_CLIP:
            lo, hi = np.percentile(pooled, FIG_CLIP)
        elif pooled:
            lo, hi = min(pooled), max(pooled)
        else:
            lo, hi = 0.0, 1.0
        trimmed = [[v for v in d if lo <= v <= hi] for d in data]

        drawable = [(xi, full, cut, c) for xi, full, cut, c in zip(x, data, trimmed, bases)
                    if len(cut) >= 2 and float(np.std(cut)) > 0]
        if drawable:
            parts = ax.violinplot([cut for _, _, cut, _ in drawable],
                                  positions=[xi for xi, _, _, _ in drawable],
                                  widths=0.82, showextrema=False, showmedians=False)
            for body, (_, _, _, c) in zip(parts['bodies'], drawable):
                body.set_facecolor(_mix(c, 0.70))
                body.set_edgecolor(c)
                body.set_alpha(1.0)
                body.set_linewidth(0.8)
                body.set_zorder(2)

        if drawable and key not in FIG_NO_BOX:
            bp = ax.boxplot([full for _, full, _, _ in drawable],
                            positions=[xi for xi, _, _, _ in drawable],
                            widths=0.13, showfliers=False, patch_artist=True, whis=1.5,
                            medianprops=dict(color='#111111', linewidth=1.1),
                            whiskerprops=dict(color='#111111', linewidth=0.7),
                            capprops=dict(linewidth=0),
                            boxprops=dict(linewidth=0.7, edgecolor='#111111'))
            for patch, (_, _, _, c) in zip(bp['boxes'], drawable):
                patch.set_facecolor(_mix(c, 0.18))
                patch.set_zorder(4)
            for art in bp['whiskers'] + bp['medians']:
                art.set_zorder(5)

        pad = 0.05 * (hi - lo) if hi > lo else 1.0
        ax.set_ylim(lo - pad, hi + pad)
        if key in FIG_NO_CLIP:                       # an integer count deserves integer ticks
            ax.set_yticks(np.arange(int(np.floor(lo)), int(np.ceil(hi)) + 1))

        ax.set_title(title, fontsize=15, pad=8)
        ax.set_ylabel(ylab, fontsize=11.5)
        ax.set_xlim(0.4, len(keep) + 0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
        ax.tick_params(axis='both', labelsize=9.5)
        # The reference and our own row are the two a reader looks for first; bolding their tick
        # label is the direct-label half of the colour encoding, so neither depends on hue alone.
        for tick, r in zip(ax.get_xticklabels(), keep):
            if r['id'] == 'reference' or r['id'] in OURS_IDS:
                tick.set_fontweight('bold')
        for sp in ax.spines.values():
            sp.set_linewidth(0.9)
            sp.set_color('#333333')

    fig.tight_layout(pad=1.4, w_pad=2.0, h_pad=2.2)
    fig.savefig(out_png, bbox_inches='tight', facecolor='white')
    fig.savefig(out_pdf, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out_png


def load_for_figure(out_dir):
    """Rebuild just enough of `results` from the written outputs to redraw the figure.

    Measuring the whole roster costs ~15 min; redrawing costs seconds. Without this, every
    tweak to a tick label would re-read 1 GB of .pt files. Only the fields make_figure touches
    are reconstructed -- this is a plotting shortcut, not a second source of truth.
    """
    def f(x):
        return None if x in ('', None) else float(x)

    over = {}
    with open(os.path.join(out_dir, 'property_overall.csv'), newline='', encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            over[r['id']] = r
    per_mol = {}
    with gzip.open(os.path.join(out_dir, 'property_per_molecule.csv.gz'), 'rt',
                   encoding='utf-8') as fh:
        rd = csv.reader(line for line in fh if not line.startswith('#'))
        next(rd)
        for row in rd:
            per_mol.setdefault(row[0], []).append([int(row[1]), row[2], int(row[3])]
                                                  + [f(v) for v in row[4:]])
    return [{'id': mid, 'label': r['label'], 'per_mol': per_mol.get(mid, [])}
            for mid, r in over.items()]


# ------------------------------------------------------------------ driver
def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--out_dir', default='results/comparison/f1_sbdd/properties')
    p.add_argument('--registry', default='configs/models.json')
    p.add_argument('--sampling_dir', default='results/sampling_results')
    p.add_argument('--canonical', default='targetdiff_vina_docked.pt',
                   help='grouped-by-pocket file that fixes the 0..99 pocket order')
    p.add_argument('--f1_csv', default=F1_CSV, help='table the gates check against')
    p.add_argument('--ids', default=None,
                   help='comma-separated registry ids; default is the full F1-F4 roster')
    p.add_argument('--figure_ids', default=None,
                   help='comma-separated ids for the figure; default is the 9-row paper roster')
    p.add_argument('--jobs', type=int, default=1, help='models measured in parallel')
    p.add_argument('--no_figure', action='store_true')
    p.add_argument('--fig_dir', default='paper/figures/unused_figures',
                   help='where the figure is written; the tables stay in --out_dir')
    p.add_argument('--refigure', action='store_true',
                   help='redraw the figure from an existing --out_dir without re-measuring')
    p.add_argument('--allow_gate_failure', action='store_true',
                   help='write the outputs even if a gate fails. Off by default on purpose: a '
                        'property table that cannot reproduce the published QED/SA/Div/n is '
                        'measuring a different population than its header claims.')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)
    o = lambda n: os.path.join(args.out_dir, n)                          # noqa: E731
    fo = lambda n: os.path.join(args.fig_dir, n)                         # noqa: E731

    if args.refigure:
        results = load_for_figure(args.out_dir)
        fig_ids = ([s.strip() for s in args.figure_ids.split(',')] if args.figure_ids
                   else [i for i in PAPER_ROSTER if any(r['id'] == i for r in results)])
        print(make_figure(results, fo('property_distributions.png'),
                          fo('property_distributions.pdf'), fig_ids))
        return

    ids = [s.strip() for s in args.ids.split(',')] if args.ids else None
    roster = build_roster(args.registry, ids)

    jobs, missing = [], []
    for m in roster:
        src = model_registry.source_for(m, 'sbdd')
        if src and os.path.isfile(src):
            jobs.append((m, src))
        else:
            missing.append((m['id'], src))
    if missing:
        print(f'[warn] {len(missing)} registered model(s) have no source file on disk:')
        for mid, src in missing:
            print(f'         {mid}: {src}')

    canon = bct.load_pt(os.path.join(args.sampling_dir, args.canonical))
    idx2name, full2idx, dir2idx = bct.build_pocket_maps(bct.canonical_index(canon))
    del canon
    print(f'Canonical pockets: {len(idx2name)}')
    print(f'Roster: {len(jobs)} models -> {args.out_dir}')

    payload = [(m, src, full2idx, dir2idx, idx2name) for m, src in jobs]
    t0 = time.time()
    if args.jobs > 1:
        import multiprocessing as mp
        with mp.get_context('fork').Pool(args.jobs) as pool:
            results = pool.map(measure_one, payload, chunksize=1)
    else:
        results = [measure_one(j) for j in payload]
    order = {m['id']: i for i, (m, _) in enumerate(jobs)}
    results.sort(key=lambda r: order[r['id']])
    for r in results:
        print(f'  {r["id"]:16s} pockets={r["n_pockets"]:3d} mols={r["n_mols"]:6d} '
              f'({r["secs"]:.0f}s)')
    print(f'measured in {time.time() - t0:.0f}s')

    stamp = (f'Built {time.strftime("%Y-%m-%d %H:%M")}; '
             f'RDKit {rdkit.__version__}; registry {args.registry}; '
             f'{len(results)} models, {sum(r["n_mols"] for r in results)} molecules')

    gate_lines, ok = run_gates(results, read_f1(args.f1_csv), args.f1_csv)
    print('\n'.join(gate_lines[-3:]))
    if not ok and not args.allow_gate_failure:
        print('\n'.join(gate_lines))
        sys.exit('ABORT: a validation gate failed; nothing was written. '
                 'Re-run with --allow_gate_failure only after reading the report above.')

    write_overall_csv(o('property_overall.csv'), results)
    write_per_target_csv(o('property_per_target.csv'), results)
    write_per_molecule(o('property_per_molecule.csv.gz'), results, stamp)
    write_txt(o('property_tables.txt'), results, stamp, gate_lines, args.f1_csv)

    written = [o(n) for n in ('property_overall.csv', 'property_per_target.csv',
                              'property_per_molecule.csv.gz', 'property_tables.txt')]
    if not args.no_figure:
        fig_ids = ([s.strip() for s in args.figure_ids.split(',')] if args.figure_ids
                   else [i for i in PAPER_ROSTER if any(r['id'] == i for r in results)])
        if make_figure(results, fo('property_distributions.png'),
                       fo('property_distributions.pdf'), fig_ids):
            written += [fo('property_distributions.png'), fo('property_distributions.pdf')]

    print('\nWrote:')
    for w in written:
        print(f'  {w}')


if __name__ == '__main__':
    main()
