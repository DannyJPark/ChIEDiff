#!/usr/bin/env python3
"""Ring size distribution panel for the F1 SBDD roster: 3- to 9-membered ring proportions.

WHY THIS EXISTS. results/comparison/f1_sbdd/comparison_overall.csv reports QED/SA/HA%/Div but no
ring-size distribution, which TargetDiff, KGDiff (results/papers/KGDiff_supplementary.pdf, Table
S1, p.3) and SGEDiff (results/papers/SGEDiff_*.pdf, Fig. 4, p.10) all report for their generated
molecules. This builder is the missing cross-model table + a Fig. 4-style figure.

CORRECTION (2026-09-11, user-flagged). The first version of this builder used a single metric --
percent of molecules containing >= 1 ring of size s -- because that is what this repo's OWN
scripts/evaluate_diffusion.py:print_ring_ratio already computes, and it is what SGEDiff's Fig. 4
caption describes. It reproduced NEITHER TargetDiff's NOR KGDiff's published numbers (L1 error
39-120 percentage points across the 7 bins), even though our targetdiff/kgdiff .pt files ARE those
papers' own released generations, unmodified -- so a mismatch there can only be a metric bug on
our side, not a different sample. KGDiff_supplementary.pdf Table S1's 6 model columns each sum to
EXACTLY 100.0% across ring sizes 3-9 (verified below); a "contains >=1 ring" metric cannot sum to
100% because one molecule with a 5- and a 6-membered ring is counted in both bins. Table S1's real
definition is per-RING, not per-molecule: (# rings of size s across all molecules) / (total # of
rings across all molecules) x 100, where "total" means rings of size 3-9 ONLY (see the second
correction below).

SGEDiff's Fig. 4 is different again: reading its printed bar values off the chart (e.g. TargetDiff:
~0.07+0.39+0.68+0.20+0.09+0.04 =~ 1.47), its 7 bars per model do NOT sum to 1.0 either -- so SGEDiff
independently uses the same non-exclusive "contains >=1 ring" convention this repo's own
print_ring_ratio does, just not KGDiff Table S1's convention. Two different, both legitimate,
metrics exist in the literature under the same "ring size distribution" name. This builder now
reports BOTH, clearly labelled, with the KGDiff-matching one (ring{s}_occ_pct) as primary because
it is the one independently checkable against a published numeric table, and the figure plots
THAT one (still in SGEDiff's panel layout -- the layout and the metric are orthogonal choices).
ring{s}_mol_pct (the original metric) is kept as a secondary column for anyone who wants the
SGEDiff/print_ring_ratio-comparable view; it is not gated against anything since no numeric table
for it exists to check against.

CONFIRMED AGAINST THE OFFICIAL KGDIFF REPO (2026-09-11, user-requested diff). Compared
/home/ktori1361/KGDiff/scripts/evaluate_diffusion.py against this repo's copy of the same file:
`print_ring_ratio` is byte-identical in both (only comments/formatting differ elsewhere in the
file). That confirms ring{s}_mol_pct is not a guess at what print_ring_ratio does -- it IS what
that shipped, public function computes, verified against its actual source. But print_ring_ratio's
own output does NOT reproduce Table S1 either, even after forcing it to sum to 100 by dividing
each bin by the sum of all 7 (L1 7.4-29.9, worse than doing nothing): {reference: 13.9, ligan: 7.4,
ar: 17.2, pocket2mol: 29.9, targetdiff: 17.2, kgdiff: 18.4} vs ring{s}_occ_pct's 2.2-5.1. So
Table S1's number was NOT produced by evaluate_diffusion.py -- confirmed below, it comes from a
second, separate script.

PROVEN (not just best-fit) AGAINST THE PAPER'S OWN REPRODUCTION NOTEBOOK (2026-09-11,
github.com/CMACH508/KGDiff/blob/main/reproduction.ipynb, user-supplied). This notebook is what
actually produced Table S1 -- its printed cell outputs for Reference/AR/Pocket2Mol/"Ours" match
the PDF table to rounding (TargetDiff's S1 column is its `rep_targetdiff` re-run; see below). Its
ring function:

    def print_ring(ring_size):                        # ring_size is a pooled Counter
        ring_res = [ring_size.get(i, 0) for i in range(3, 10)]
        ring_res = np.array(ring_res) / np.sum(ring_res)          # <- divides by the 3-9 total
        for i in range(3, 10):
            print('%d ring | %.2f%%' % (i, ring_res[i - 3] * 100))

    ring_size = Counter({})
    for r in results:
        ring_size.update(r['chem_results']['ring_size'])          # Counter.update() ADDS counts

`Counter.update()` on a Counter argument adds counts rather than overwriting keys, so `ring_size`
ends up holding the TOTAL ring-occurrence count per size across the whole population, and
`print_ring` divides each bin by the sum of all of them -- exactly ring{s}_occ_pct's definition,
not a coincidental best fit. ring{s}_mol_pct (print_ring_ratio's own metric) never appears in this
notebook at all.

SECOND CORRECTION (2026-09-11, same day). The first occ% divided the 3-9 counts by the count of
ALL rings, macrocycles included, so rows did not sum to 100 (KGDiff 94.9, PIDiff 94.1, test set
98.0) and the L1 against Table S1 read 0.3-5.1. That residual was put down to "RDKit-version
ring-perception drift", and that explanation was WRONG. `print_ring` above builds its denominator
from `ring_res`, i.e. from sizes 3-9 only. Fixing the denominator alone took KGDiff from 5.10 to
0.22 L1 and the test set from 2.20 to 0.24. Rings of 10+ atoms are now a separate column
(ring_ge10_share_pct, as a share of ALL rings) rather than an invisible hole in the 3-9 sum.

Every recomputed ring Counter matches the file's stored one count-for-count, on every model.
The only exception is a defect in one source file, and it is the only real difference between
this builder and the notebook: 1,036 of AR's 9,295 molecules store an EMPTY ring_size although
the molecule has rings (1,810 three-membered rings among them). The notebook sums the stored
value, so KGDiff's published AR column undercounts those rings; this builder recomputes them. So
AR is the one row where our occ% deliberately departs from the paper: 3-ring 31.1 vs 29.6,
6-ring 49.7 vs 51.4 (L1 4.39 against the notebook), and ours is the correct count. The old G1 skipped these molecules because
`if stored` treats an empty Counter as missing. G1 now compares full counts, empty included,
and reports this class as `stale` instead of silently dropping it.

G0 is now checked against the notebook's own printed cell outputs, on the SAME files, rather than
against Table S1. S1's TargetDiff column comes from the notebook's `rep_targetdiff` (KGDiff's own
re-run, benchmark/repeat_targetdiff.pt, which we do not have), not from
targetdiff_vina_docked.pt. The notebook printed both files, and the one we hold matches its cell
17 output to the second decimal. S1 is kept as NOTEBOOK-vs-PAPER context only.

MEASURED, NOT COPIED. Same discipline as build_property_tables.py: 16 of 17 .pt files already
carry chem_results['ring_size'], but each was computed by that model's own generation-time RDKit
-- a different pipeline, and for some baselines a different RDKit version, per molecule. Every
value here is instead recomputed from the stored RDKit `mol` under ONE interpreter, which also
fills MolCRAFT's ring sizes (its chem_results carries only qed/sa, no ring_size). Gate G1 checks
the recomputation against what each file stored (for the 16 that store one), so "recomputed" is a
verified claim, not an assumption. (G1 needed its own fix: our own vina_fixed_best/noguide files
store ring_size with STRING keys -- {'6': 3} -- because of a JSON merge step elsewhere in our
docking pipeline that baseline files never went through; comparing against int keys made G1 FAIL
on ~every one of our own molecules even though the counts agreed exactly. Fixed by normalising
stored keys to int before comparing.)

POPULATION. Docking-successful molecules only -- the repo-wide rule (results/comparison/README.md,
"no exceptions"), matching every other F1-F4 table including build_property_tables.py. This is a
no-op for every baseline file (bct.docked_only's own docstring: they ship pre-filtered to
docking-successful already, confirmed again here -- "ALL valid" and "docked-only" gave IDENTICAL
occ_pct for every baseline during the metric-bug investigation). For OUR rows it is a survivor
population: the 15-28% fragmented share never docks and is absent here.

NO TrainingData BAR. SGEDiff's Fig. 4 also plots the CrossDocked *training*-set ligand ring
distribution. This repo has no pre-extracted training-set ligand cache (checked: no *train*.pt
under data/ or results/), so the figure below carries only the "reference" row (this repo's
100-pocket test-set crystal ligands) as the ground-truth anchor -- SGEDiff's "RefData" equivalent,
not its "TrainingData".

ENVIRONMENT. Run under `kgdiff` (RDKit 2024.03.6), same as build_property_tables.py. NOT
`posecheck`: CLAUDE.md pins that env's RDKit 2026.3.4 and requires it not leak into kgdiff.

Usage:
    conda run -n kgdiff python scripts/build_ring_size_table.py --jobs 8
    conda run -n kgdiff python scripts/build_ring_size_table.py --ids reference,targetdiff,ours_vina
"""

import argparse
import csv
import os
import sys
import time
from collections import Counter

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)     # build_comparison_tables, model_registry, build_property_tables
sys.path.insert(0, _ROOT)     # utils.evaluation

from rdkit import RDLogger                                               # noqa: E402
import rdkit                                                             # noqa: E402

RDLogger.DisableLog('rdApp.*')

import build_comparison_tables as bct                                    # noqa: E402
import model_registry                                                    # noqa: E402
from build_property_tables import (sanitized_copy, _clean, build_roster,  # noqa: E402
                                    read_f1, F1_CSV, PAPER_ROSTER, PAPER_RELABEL,
                                    C_REF, C_BASE, C_OURS, OURS_IDS, _mix)

RING_SIZES = list(range(3, 10))     # 3..9: KGDiff Table S1's rows AND SGEDiff Fig.4's 7 panels

# results/papers/KGDiff_supplementary.pdf, page 3, "Table S1. Ring size distributions of the
# generated molecules from each baseline and our model." Transcribed verbatim; each column sums
# to 100.0. CONTEXT ONLY, not gated: G0 checks KGDIFF_NOTEBOOK below, which was printed from the
# exact files we hold. S1 differs from it by rounding, except that its TargetDiff column is the
# notebook's rep_targetdiff re-run, and its "liGAN" is the notebook's cvae_path =
# CVAE_test_docked_sf1.5.pt (registry 'cvae', not 'ligan').
PUBLISHED_KGDIFF_S1 = {
    'reference':  {3: 1.7, 4: 0.0, 5: 30.2, 6: 67.4, 7: 0.7, 8: 0.0, 9: 0.0},
    'ligan':      {3: 28.1, 4: 15.7, 5: 29.8, 6: 22.7, 7: 2.6, 8: 0.8, 9: 0.3},
    'ar':         {3: 29.6, 4: 0.0, 5: 16.1, 6: 51.4, 7: 1.7, 8: 0.7, 9: 0.5},
    'pocket2mol': {3: 0.1, 4: 0.0, 5: 16.4, 6: 80.4, 7: 2.6, 8: 0.3, 9: 0.1},
    'targetdiff': {3: 0.0, 4: 2.8, 5: 30.5, 6: 51.3, 7: 12.0, 8: 2.6, 9: 0.8},
    'kgdiff':     {3: 0.0, 4: 2.9, 5: 21.3, 6: 46.0, 7: 21.2, 8: 6.4, 9: 2.2},
}
# github.com/CMACH508/KGDiff reproduction.ipynb, printed outputs of print_results() on the SAME
# files this builder reads (cells 10/13/15/17/23: crossdocked_test / ar / pocket2mol / targetdiff
# / our_vina_score_docked.pt), two decimals as printed. Gate G0 reproduces these from each file's
# STORED ring_size with the notebook's formula, which checks the formula rather than the RDKit.
KGDIFF_NOTEBOOK = {
    'reference':  {3: 1.65, 4: 0.00, 5: 30.17, 6: 67.36, 7: 0.83, 8: 0.00, 9: 0.00},
    'ar':         {3: 29.60, 4: 0.04, 5: 16.10, 6: 51.44, 7: 1.64, 8: 0.69, 9: 0.49},
    'pocket2mol': {3: 0.12, 4: 0.02, 5: 16.38, 6: 80.41, 7: 2.61, 8: 0.35, 9: 0.12},
    'targetdiff': {3: 0.00, 4: 2.79, 5: 30.78, 6: 50.73, 7: 12.13, 8: 2.68, 9: 0.88},
    'kgdiff':     {3: 0.00, 4: 2.92, 5: 21.29, 6: 46.02, 7: 21.17, 8: 6.37, 9: 2.23},
}
# Printed at 2 dp, so a correct reproduction differs by at most 0.005 per bin.
G0_MAX_ABS_TOLERANCE = 0.006


# ------------------------------------------------------------------ per-molecule extraction
def extract_ring_props(mol, need_fp=False):
    """Extractor handed to build_comparison_tables.normalize_model.

    Mirrors build_property_tables.extract_props: starts from extract_mol (pocket alignment,
    docked-only filter, stored qed/sa) and adds a freshly recomputed ring_sizes Counter, keeping
    the file's own stored value under stored_rings purely to feed gate G1.
    """
    md = bct.extract_mol(mol, need_fp)
    chem = mol.get('chem_results') or {}
    stored = chem.get('ring_size')
    # Keys are int in every baseline .pt (a live Counter(int, int)) but STRING in our own
    # vina_fixed_best / noguide files -- those went through a JSON merge step somewhere in
    # our docking pipeline that baselines never did, which stringifies dict keys. Without this
    # normalisation {'6': 3} != {6: 3} even though the ring counts agree exactly, which is a
    # gate false-positive on ~every one of our own molecules, not a real recomputation drift.
    # `is not None`, not truthiness: an EMPTY stored Counter is a real value (an acyclic molecule,
    # or AR's stale entries), and treating it as missing is what hid AR's 1,036 defects from G1.
    md['stored_rings'] = {int(k): v for k, v in stored.items()} if stored is not None else None

    md['ring_sizes'] = None
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

    try:
        md['ring_sizes'] = dict(Counter(len(r) for r in sane.GetRingInfo().AtomRings()))
        md['calc_ok'] = True
    except Exception:
        md['calc_ok'] = False
    return md


# ------------------------------------------------------------------ aggregation
def ring_mol_pcts(mds):
    """({ring_size: pct of MOLECULES with >=1 ring of that size}, n_measured). Secondary metric --
    non-exclusive (rows need not sum to 100), matches print_ring_ratio and (by inspection) SGEDiff
    Fig. 4."""
    have = [md['ring_sizes'] for md in mds if md['ring_sizes'] is not None]
    n = len(have)
    if n == 0:
        return {s: None for s in RING_SIZES}, 0
    return {s: 100.0 * sum(1 for rs in have if rs.get(s, 0) > 0) / n for s in RING_SIZES}, n


def occ_from_counters(counters):
    """KGDiff reproduction.ipynb print_ring, verbatim in effect: pool ring counts across molecules,
    then divide each 3-9 bin by the 3-9 total. Rings of 10+ atoms are OUTSIDE the denominator, so
    the 7 bins sum to 100 exactly. Returns ({size: pct}, total_3_9, total_all)."""
    pooled = Counter()
    for rs in counters:
        pooled.update(rs)
    total_3_9 = sum(pooled.get(s, 0) for s in RING_SIZES)
    total_all = sum(pooled.values())
    if total_3_9 == 0:
        return {s: None for s in RING_SIZES}, 0, total_all
    return ({s: 100.0 * pooled.get(s, 0) / total_3_9 for s in RING_SIZES}, total_3_9, total_all)


def ring_occ_pcts(mds):
    """({ring_size: pct of 3-9 RING OCCURRENCES of that size}, n_measured, total_3_9, total_all).
    Primary metric -- matches KGDiff's reproduction notebook / Table S1; sums to 100 across
    RING_SIZES (a molecule with two 6-membered rings contributes 2 to the 6-ring count and 2 to
    the denominator, not 1)."""
    have = [md['ring_sizes'] for md in mds if md['ring_sizes'] is not None]
    occ, total_3_9, total_all = occ_from_counters(have)
    return occ, len(have), total_3_9, total_all


def measure_one(job):
    """Load one model's .pt and reduce it to plain picklable data."""
    entry, src, full2idx, dir2idx, idx2name = job
    mid, label = entry['id'], entry['label']
    t0 = time.time()

    obj = bct.load_pt(src)
    per_pocket, flat = bct.normalize_model(obj, full2idx, dir2idx,
                                           need_fp=False, extract=extract_ring_props)
    del obj

    valid = bct.docked_only(flat)
    pooled_mol_pct, n_ring_measured = ring_mol_pcts(valid)
    pooled_occ_pct, _, total_rings, total_rings_all = ring_occ_pcts(valid)
    # Over EVERY measured molecule, acyclic ones included as 0. The first version filtered on
    # `if md['ring_sizes']`, which drops the empty dict, so "mean/mol" was really "mean per
    # ring-bearing molecule" (test set 2.84 where 247 rings / 100 molecules is 2.47).
    n_rings = _clean([sum(md['ring_sizes'].values()) for md in valid
                      if md['ring_sizes'] is not None])
    mean_rings_per_mol = float(np.mean(n_rings)) if n_rings else None
    acyclic_pct = (100.0 * sum(1 for v in n_rings if v == 0) / len(n_rings)) if n_rings else None
    ge10_share = (100.0 * (total_rings_all - total_rings) / total_rings_all
                  if total_rings_all else None)

    # G0 input: the notebook's formula on each file's STORED ring_size, over the same population.
    # Only defined when every measured molecule stored one (MolCRAFT stores none).
    stored = [md['stored_rings'] for md in valid if md['ring_sizes'] is not None]
    stored_occ_pct = (occ_from_counters(stored)[0]
                      if stored and all(s is not None for s in stored) else None)

    per_target = []
    for i in sorted(per_pocket):
        pk_valid = bct.docked_only(per_pocket[i])
        if not pk_valid:
            continue
        mol_pct, n = ring_mol_pcts(pk_valid)
        occ_pct, _, _, _ = ring_occ_pcts(pk_valid)
        row = {'pocket_idx': i, 'pocket_name': idx2name.get(i, ''), 'n_mol': len(pk_valid),
               'n_ring_measured': n}
        row.update({f'ring{s}_occ_pct': occ_pct[s] for s in RING_SIZES})
        row.update({f'ring{s}_mol_pct': mol_pct[s] for s in RING_SIZES})
        per_target.append(row)

    # G1: recomputed ring Counter vs each file's own stored chem_results['ring_size'], FULL counts
    # over every size, empty stored Counters included (every model but MolCRAFT stores one).
    # `stale` = stored empty while the molecule has rings: a defect in the SOURCE file (AR's
    # generation-time pipeline), reported but not fatal. Any other disagreement is fatal.
    compared = [(md['ring_sizes'], md['stored_rings']) for md in valid
               if md['ring_sizes'] is not None and md['stored_rings'] is not None]
    stale = sum(1 for new, old in compared if not old and new)
    mismatches = sum(1 for new, old in compared if old and new != old)

    gate = {
        'n_mols': len(valid), 'n_pockets': len(per_pocket),
        'n_ring_measured': n_ring_measured,
        'n_unsanitizable': sum(1 for md in valid if md.get('unsanitizable')),
        'n_calc_fail': sum(1 for md in valid
                           if md['calc_ok'] is False and not md.get('unsanitizable')),
        'n_compared_to_stored': len(compared),
        'n_mismatch_vs_stored': mismatches,
        'n_stale_stored': stale,
    }

    return {'id': mid, 'label': label, 'source': src,
            'n_pockets': len(per_pocket), 'n_mols': len(valid),
            'pooled_occ_pct': pooled_occ_pct, 'pooled_mol_pct': pooled_mol_pct,
            'stored_occ_pct': stored_occ_pct,
            'total_rings': total_rings, 'total_rings_all': total_rings_all,
            'ring_ge10_share_pct': ge10_share, 'acyclic_pct': acyclic_pct,
            'mean_rings_per_mol': mean_rings_per_mol,
            'per_target': per_target, 'gate': gate, 'secs': time.time() - t0}


# ------------------------------------------------------------------ validation gates
def _cell(row, key):
    v = (row or {}).get(key, '')
    return v.strip() if isinstance(v, str) else v


def run_gates(results, f1, f1_path):
    """G0-G2. Returns (lines, ok). A failure is fatal, same policy as build_property_tables.py.

    G0 checks the FORMULA: the notebook's print_ring applied to each file's STORED ring_size must
    reproduce the notebook's own printed output on the same file (KGDIFF_NOTEBOOK) to print
    rounding. It runs on the stored values, not the recomputed ones, because the notebook read the
    stored values. That is what makes the check exact rather than toleranced. `L1 recomp` then
    shows how far the published number sits from OUR measurement. It is informational, and it is
    0 up to rounding everywhere except AR, whose stored values are stale (see G1).

    G1 has no tolerance: ring counts are discrete and come from the SAME rdkit mol object, so any
    disagreement other than the known stale-empty class means the recomputation and the file's own
    generation-time RDKit differ on the same molecule, and that is worth stopping on.
    """
    L = ['GATE REPORT', '=' * 110, '',
         'Reference (G0): KGDiff reproduction.ipynb printed cell outputs (same .pt files)',
         f'Reference (G2): {f1_path}',
         '  G0  notebook formula on each file\'s STORED ring_size reproduces the notebook\'s printed',
         f'      output to max |diff| <= {G0_MAX_ABS_TOLERANCE} pp per bin (2-dp print rounding), for the',
         f'      {len(KGDIFF_NOTEBOOK)} models the notebook printed. `L1 recomp` = our recomputed occ% vs'
         ' that output.',
         '  G1  recomputed ring Counter equals each file\'s stored chem_results[\'ring_size\'] count-',
         '      for-count, empty Counters included. `stale` (stored empty, molecule has rings) is a',
         '      source-file defect and is reported, not fatal.',
         '  G2  n_mols / n_pockets equal the F1 cell exactly (same population, same alignment)',
         '']
    hdr = ['Model', 'G0 max|d|', 'L1 recomp', 'G1 ring', 'G2 n', 'compared', 'mismatch',
           'stale', 'unsan', 'calc err']
    rows, ok = [], True
    for r in results:
        g, ref = r['gate'], f1.get(r['id'])
        nb = KGDIFF_NOTEBOOK.get(r['id'])
        if nb is None:
            g0, l1r = 'n/a', 'n/a'
        elif r['stored_occ_pct'] is None:
            g0, l1r = 'FAIL no stored', 'n/a'
        else:
            dmax = max(abs(r['stored_occ_pct'][s] - nb[s]) for s in RING_SIZES)
            g0 = f'{dmax:.4f}' if dmax <= G0_MAX_ABS_TOLERANCE else f'FAIL {dmax:.4f}'
            l1r = f'{sum(abs((r["pooled_occ_pct"][s] or 0.0) - nb[s]) for s in RING_SIZES):.2f}'
        g1 = 'n/a' if g['n_compared_to_stored'] == 0 else (
            'PASS' if g['n_mismatch_vs_stored'] == 0 else f'FAIL {g["n_mismatch_vs_stored"]}')
        if ref is None:
            g2 = 'n/a'
        else:
            nbad = [f'{k} {g[k]} vs {_cell(ref, k)}'
                    for k in ('n_mols', 'n_pockets') if str(g[k]) != _cell(ref, k)]
            g2 = 'PASS' if not nbad else 'FAIL ' + '; '.join(nbad)
        cells = [r['id'], g0, l1r, g1, g2, g['n_compared_to_stored'], g['n_mismatch_vs_stored'],
                 g['n_stale_stored'], g['n_unsanitizable'], g['n_calc_fail']]
        if any(str(c).startswith('FAIL') for c in cells):
            ok = False
        rows.append(cells)
    L += bct.render_table(hdr, rows, ['l'] + ['l'] * (len(hdr) - 1))
    L += ['', f'GATES: {"ALL PASS" if ok else "FAILED"}', '',
          'G0       n/a for models the notebook never printed: every other baseline, and all of our',
          '         own rows, have no external numeric output to check against.',
          'L1 recomp  AR is the one expected non-zero: its source file stores an EMPTY ring_size for',
          '         molecules that have rings (`stale`), so the notebook -- and KGDiff Table S1 --',
          '         undercount them. Our recomputed value is the correct one.',
          'stale    stored ring_size empty although the molecule has rings. Source-file defect.',
          'unsan    molecules inside the docking-successful set that fail RDKit sanitization; no',
          '         ring_sizes value, excluded from the ring-size denominator (n_ring_measured).',
          'calc err ring perception still raised after the sanitized-copy retry. Expected: 0.',
          'compared molecules where this file stored its own chem_results[\'ring_size\'], empty',
          '         included (MolCRAFT stores none, so its "compared" is 0 and G1 reads n/a).']
    return L, ok


# ------------------------------------------------------------------ writers
def _num(x, nd):
    return '' if x is None else f'{x:.{nd}f}'


def write_overall_csv(path, results):
    # total_rings is the 3-9 count, i.e. the occ% denominator; total_rings_all adds the 10+ rings.
    cols = (['id', 'label', 'n_pockets', 'n_mols', 'n_ring_measured', 'total_rings',
            'total_rings_all', 'ring_ge10_share_pct', 'acyclic_pct', 'mean_rings_per_mol']
           + [f'ring{s}_occ_pct' for s in RING_SIZES]
           + [f'ring{s}_mol_pct' for s in RING_SIZES] + ['source'])
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in results:
            row = {'id': r['id'], 'label': r['label'], 'n_pockets': r['n_pockets'],
                   'n_mols': r['n_mols'], 'n_ring_measured': r['gate']['n_ring_measured'],
                   'total_rings': r['total_rings'], 'total_rings_all': r['total_rings_all'],
                   'ring_ge10_share_pct': _num(r['ring_ge10_share_pct'], 1),
                   'acyclic_pct': _num(r['acyclic_pct'], 1), 'source': r['source'],
                   'mean_rings_per_mol': _num(r['mean_rings_per_mol'], 3)}
            row.update({f'ring{s}_occ_pct': _num(r['pooled_occ_pct'][s], 1) for s in RING_SIZES})
            row.update({f'ring{s}_mol_pct': _num(r['pooled_mol_pct'][s], 1) for s in RING_SIZES})
            w.writerow([row[c] for c in cols])


def write_per_target_csv(path, results):
    cols = (['pocket_idx', 'pocket_name', 'id', 'label', 'n_mol']
           + [f'ring{s}_occ_pct' for s in RING_SIZES] + [f'ring{s}_mol_pct' for s in RING_SIZES])
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in results:
            for t in r['per_target']:
                row = {'pocket_idx': t['pocket_idx'], 'pocket_name': t['pocket_name'],
                       'id': r['id'], 'label': r['label'], 'n_mol': t['n_mol']}
                row.update({f'ring{s}_occ_pct': _num(t[f'ring{s}_occ_pct'], 1) for s in RING_SIZES})
                row.update({f'ring{s}_mol_pct': _num(t[f'ring{s}_mol_pct'], 1) for s in RING_SIZES})
                w.writerow([row[c] for c in cols])


def txt_header(stamp, f1_path):
    return [
        '=' * 130,
        'RING SIZE DISTRIBUTION -- ring sizes 3..9',
        '=' * 130,
        '',
        stamp,
        '',
        'A SUB-TABLE of F1, not a replacement. results/comparison/f1_sbdd/comparison_tables.txt',
        'stays the F1 table of record; these are the SAME molecules under the SAME population',
        'rule, reporting the ring-size panel TargetDiff/KGDiff/SGEDiff all report and F1 does not.',
        '',
        'POPULATION. Docking-successful molecules only -- the repo-wide rule. For OUR rows this is',
        'a SURVIVOR population: the 15-28% fragmented share never docks and is absent from these',
        'percentages, the same caveat F1 documents for HA%.',
        '',
        'TWO METRICS, both reported, see file docstring for the full derivation:',
        '  occ%  PRIMARY. % of RING OCCURRENCES of size s over the count of 3-9 rings. Sums to 100',
        '        across the 7 columns. This is KGDiff reproduction.ipynb print_ring, the source of',
        '        KGDiff Table S1 (gate G0 reproduces its printed output on every run). A molecule',
        '        with two 6-membered rings contributes 2 to ring6 and 2 to the denominator. Rings of',
        '        10+ atoms sit outside that denominator, and "10+ %all" reports them as a share of',
        '        ALL rings.',
        '  mol%  SECONDARY. % of MOLECULES containing >= 1 ring of size s. Non-exclusive, rows do',
        '        NOT sum to 100. Matches scripts/evaluate_diffusion.py:print_ring_ratio and (by',
        '        inspection of its printed bar values) SGEDiff Fig. 4 -- a different, also-',
        '        published convention, with no numeric table of its own to gate against.',
        '',
        'FORMAT REFERENCE. results/papers/SGEDiff_*.pdf, Fig. 4 (p.10): 7 bar-chart panels, ring',
        'sizes 3-9, one bar per model plus TrainingData and RefData. ring_size_distribution.png',
        'reuses that PANEL LAYOUT for occ% (the metric and the layout are independent choices); we',
        'have no TrainingData analogue (see file docstring), so only reference (test-set crystal',
        'ligands) anchors it.',
        '',
        'mean/mol is over every measured molecule, acyclic ones counted as 0; acyc% is their share.',
        '',
        'AR DEPARTS FROM THE PUBLISHED NUMBER ON PURPOSE. Its source file stores an empty',
        'ring_size for molecules that do have rings (G1 `stale`). KGDiff summed the stored value, so',
        'its AR column undercounts those rings; the value here is recomputed and correct.',
        '',
        f'Gates below check the occ% formula against KGDiff\'s notebook output (G0) and this table\'s',
        f'population against {f1_path} (G2).',
        '',
    ]


def txt_table(results, ids=None, relabel=None, title='', note='', metric='occ'):
    key = f'pooled_{metric}_pct'
    keep = [r for r in results if ids is None or r['id'] in ids]
    if ids:
        keep.sort(key=lambda r: ids.index(r['id']))
    relabel = relabel or {}
    L = ['-' * 130, title, '-' * 130, '']
    if note:
        L += [note, '']

    hdr = (['Model', 'n_pk', 'n_mol'] + [f'{s}-ring {metric}%' for s in RING_SIZES]
           + ['10+ %all', 'acyc%', 'mean/mol'])
    rows = [[relabel.get(r['id'], r['label']), r['n_pockets'], r['n_mols']]
            + [_num(r[key][s], 1) for s in RING_SIZES]
            + [_num(r['ring_ge10_share_pct'], 1), _num(r['acyclic_pct'], 1),
               _num(r['mean_rings_per_mol'], 2)]
            for r in keep]
    L += bct.render_table(hdr, rows, ['l'] + ['r'] * (len(hdr) - 1))
    L += ['']
    return L


def write_txt(path, results, stamp, gate_lines, f1_path):
    L = txt_header(stamp, f1_path)
    L += txt_table(results, metric='occ',
                   title='OVERALL, occ% (PRIMARY -- KGDiff notebook / Table S1 convention, see header)',
                   note='Row order is the registry `order`, never a sort on the numbers.')
    L += txt_table(results, metric='mol',
                   title='OVERALL, mol% (SECONDARY -- non-exclusive, matches print_ring_ratio)')
    have = {r['id'] for r in results}
    paper = [i for i in PAPER_ROSTER if i in have]
    if len(paper) == len(PAPER_ROSTER):
        L += txt_table(results, ids=paper, relabel=PAPER_RELABEL, metric='occ',
                       title='PAPER ROSTER, occ% -- the nine rows paper/make_tables.sh prints',
                       note='Same numbers, restricted and relabelled to the manuscript roster so a\n'
                            'ring-size claim in the text can be read against one block.')
    L += gate_lines
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')


# ------------------------------------------------------------------ figure
# SGEDiff Fig. 4 layout: 3/4/5/6-membered on the top row, 7/8/9-membered on the bottom row (its
# 4th slot left empty, same as the source figure). Bars grouped by model within each ring size.
# Plots occ_pct (the metric checked against KGDiff Table S1), not mol_pct.
FIG_LAYOUT = [(3, 0, 0), (4, 0, 1), (5, 0, 2), (6, 0, 3),
             (7, 1, 0), (8, 1, 1), (9, 1, 2)]


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

    fig, axes = plt.subplots(2, 4, figsize=(15.6, 6.4), dpi=200)
    axes[1, 3].axis('off')     # SGEDiff's chart leaves this slot empty too
    x = np.arange(1, len(keep) + 1)

    for s, row, col in FIG_LAYOUT:
        ax = axes[row, col]
        vals = [r['pooled_occ_pct'][s] if r['pooled_occ_pct'][s] is not None else 0.0
               for r in keep]
        ax.bar(x, vals, width=0.72, color=[_mix(c, 0.12) for c in bases],
              edgecolor=bases, linewidth=0.9, zorder=2)
        ax.set_title(f'{s}-atom membered', fontsize=13, pad=6)
        ax.set_ylabel('Ring occurrence (%)', fontsize=10.5)
        ax.set_ylim(0, 100)
        ax.set_xlim(0.3, len(keep) + 0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8.5)
        ax.tick_params(axis='both', labelsize=8.5)
        for tick, r in zip(ax.get_xticklabels(), keep):
            if r['id'] == 'reference' or r['id'] in OURS_IDS:
                tick.set_fontweight('bold')
        ax.yaxis.grid(True, linewidth=0.5, color='#dddddd', zorder=0)
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_linewidth(0.9)
            sp.set_color('#333333')

    fig.tight_layout(pad=1.4, w_pad=1.8, h_pad=2.2)
    fig.savefig(out_png, bbox_inches='tight', facecolor='white')
    fig.savefig(out_pdf, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out_png


def load_for_figure(out_dir):
    """Rebuild just enough of `results` from ring_size_overall.csv to redraw."""
    def f(x):
        return None if x in ('', None) else float(x)

    results = []
    with open(os.path.join(out_dir, 'ring_size_overall.csv'), newline='', encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            results.append({'id': r['id'], 'label': r['label'],
                            'pooled_occ_pct': {s: f(r[f'ring{s}_occ_pct']) for s in RING_SIZES}})
    return results


# ------------------------------------------------------------------ driver
def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--out_dir', default='results/comparison/f1_sbdd/ring_size')
    p.add_argument('--registry', default='configs/models.json')
    p.add_argument('--sampling_dir', default='results/sampling_results')
    p.add_argument('--canonical', default='targetdiff_vina_docked.pt',
                   help='grouped-by-pocket file that fixes the 0..99 pocket order')
    p.add_argument('--f1_csv', default=F1_CSV, help='table this population is checked against')
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
                   help='write the outputs even if a gate fails. Off by default on purpose.')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)
    o = lambda n: os.path.join(args.out_dir, n)                          # noqa: E731
    fo = lambda n: os.path.join(args.fig_dir, n)                         # noqa: E731

    if args.refigure:
        results = load_for_figure(args.out_dir)
        fig_ids = ([s.strip() for s in args.figure_ids.split(',')] if args.figure_ids
                  else [i for i in PAPER_ROSTER if any(r['id'] == i for r in results)])
        print(make_figure(results, fo('ring_size_distribution.png'),
                          fo('ring_size_distribution.pdf'), fig_ids))
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

    write_overall_csv(o('ring_size_overall.csv'), results)
    write_per_target_csv(o('ring_size_per_target.csv'), results)
    write_txt(o('ring_size_tables.txt'), results, stamp, gate_lines, args.f1_csv)

    written = [o(n) for n in ('ring_size_overall.csv', 'ring_size_per_target.csv',
                              'ring_size_tables.txt')]
    if not args.no_figure:
        fig_ids = ([s.strip() for s in args.figure_ids.split(',')] if args.figure_ids
                  else [i for i in PAPER_ROSTER if any(r['id'] == i for r in results)])
        if make_figure(results, fo('ring_size_distribution.png'),
                       fo('ring_size_distribution.pdf'), fig_ids):
            written += [fo('ring_size_distribution.png'), fo('ring_size_distribution.pdf')]

    print('\nWrote:')
    for w in written:
        print(f'  {w}')


if __name__ == '__main__':
    main()
