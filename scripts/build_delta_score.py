#!/usr/bin/env python3
"""
Aggregate the off-target docking run into the Delta Score (selectivity) tables.

TWO SIGN CONVENTIONS, both reported, because the literature uses both:

    Delta      = Vina(on-target) - Vina(off-target)     MORE NEGATIVE = MORE SELECTIVE
    Delta_SBE  = Vina(off-target) - Vina(on-target)     HIGHER        = MORE SELECTIVE

`Delta` is the form the experiment was specified in and matches the original Delta
Score paper (arXiv 2311.12035 Eq.2). `Delta_SBE` is SBE-Diff's Eq.6
(analysis/SBE_Diff_수정.pdf), i.e. Delta Score(y_i) = 1/m_i * sum_j (-S(x_ij,y_i) + S(x_ij,y_k)),
and is the one to compare against their published Table 2 / Table 4 numbers.

Beware: the sentence directly under SBE-Diff Eq.6 says "a smaller value of Delta Score
indicates ... high affinity towards target y_i itself". That contradicts their own
equation and their own tables (Table 2 heads the column "Delta up", and Reference
scores 1.158, the best of any row). The prose is left over from their Eq.5, which is
the opposite sign. Their Table 3 settles it arithmetically: TargetDiff ori 6.665,
shuffle 6.320, delta 0.335 = 6.665 - 6.320. Higher is better.

Everything is PAIRED PER MODE: a molecule enters the statistics for mode m only if
it has both an on-target and an off-target value for that mode. score_only often
fails where dock succeeds, so the per-mode n is reported separately.

Usage:
    conda activate kgdiff
    python scripts/build_delta_score.py
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from delta_score_common import (REGISTRY, load_registry, load_assignment, load_test_pockets,
                                offtarget_out_path)
from build_comparison_tables import render_table, fmt


def report_roster(registry=REGISTRY):
    """Registry ids of the models the manuscript table shows, or [] if undeclared."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import model_registry                                                    # noqa: E402
    fam = model_registry.load(registry)['families'].get('delta') or {}
    return fam.get('report_roster') or []


def delta_tag_of(registry, model_id):
    """Registry id -> delta tag. They differ: ours_vina is tagged vina_fixed_on."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import model_registry                                                    # noqa: E402
    for m in model_registry.load(registry)['models']:
        if m['id'] == model_id:
            return (m.get('ids') or {}).get('delta_tag')
    return None

MODES = ('dock', 'minimize', 'score_only')
MODE_LABEL = {'dock': 'Vina Dock', 'minimize': 'Vina Min', 'score_only': 'Vina Score'}
W = 190
# A rigidly translated ligand whose atoms fall outside the Vina grid does NOT make Vina
# raise -- it makes Vina return the grid penalty as if it were a score (observed up to
# +4813 kcal/mol off-target, +1559 on-target). Those are box artefacts, not binding
# energies, so the reported statistics winsorise both sides at +/-20 kcal/mol.
#
# Why 20 is defensible: a drug-like ligand's real Vina score never approaches this
# magnitude (the strongest anywhere in this dataset is about -16). Measured over the
# common pocket set, the choice barely matters -- +/-100, +/-50, +/-25 and +/-20 agree to
# within ~0.05 kcal/mol per model, while going UNclipped moves AliDiff by 0.9. Below
# about +/-15 the clip starts biting real values (the Reference row begins to move).
# The threshold sweep is printed in the RAW section so the reader can check this.
# Win% and Ratio% are rank-based and immune to the artefacts by construction, so they
# are computed on RAW values -- clipping could only manufacture artificial ties.
CLIP = 20.0


def collect(cfg, asg, out_root, strict_box=False):
    """{model_tag: {pocket_idx: [molecule record, ...]}} from the docking outputs."""
    data = {}
    missing = defaultdict(list)
    for model in cfg['models']:
        tag = model['tag']
        per_pocket = {}
        for i_str in sorted(asg['mol_selection'][tag], key=int):
            i = int(i_str)
            path = offtarget_out_path(tag, i, out_root)
            if not os.path.exists(path):
                missing[tag].append(i)
                continue
            blob = torch.load(path)
            mols = blob['molecules']
            if strict_box:
                mols = [m for m in mols if not m.get('box_expanded')]
            if mols:
                per_pocket[i] = mols
        data[tag] = per_pocket
    return data, missing


def paired(mols, mode):
    """[(on, off)] for molecules with both values in this mode."""
    out = []
    for m in mols:
        on, off = m['on'].get(mode), m['off'].get(mode)
        if on is not None and off is not None:
            out.append((float(on), float(off)))
    return out


def stats(per_pocket, mode, ref=None, clip=CLIP):
    """All reported statistics for one Vina mode.

    SIGN: every Delta here is (off - on), i.e. SBE-Diff Eq.6 -- HIGHER = MORE SELECTIVE.

    clip: winsorise both sides at +/-clip kcal/mol before computing the VALUE-based
    columns (Delta_*, On_*, Off_*). Pass clip=None for the raw numbers. Win% and Ratio%
    always use raw values -- they are rank comparisons, so an artefact of +4813 and one of
    +25 rank identically, and clipping them would only create ties that did not exist.

    ONE aggregation rule for every reported column, in two stages:

      stage 1  reduce each pocket to a SINGLE number
                 Delta / On / Off -> that pocket's MEAN over its molecules
                                     (SBE-Diff Eq.6 is literally 1/m_i * sum_j, a mean;
                                      the unbiasedness argument in their Appendix A.2
                                      needs a mean, and the median would discard the
                                      genuinely selective upper tail -- 16% of our
                                      molecules sit at delta > +5 kcal/mol)
                 Ratio% / Win%    -> that pocket's rate (already a single scalar)
      stage 2  combine the per-pocket numbers across pockets
                 (Avg) -> mean    - the unbiased estimator; every target counts once
                 (Med) -> median  - robustness against the n~=1 off-target draw
                                    (measured 20-35% more stable under pocket resampling)

    This is exactly SBE-Diff Table 2's aggregation: their "Delta mean" is our (Avg) and
    their "Delta median" is our (Med), for every column alike.

    The pocket is the independent unit of this experiment: with n~ = 1 every molecule in a
    pocket faces the SAME off-target protein, so molecules within a pocket are correlated
    by construction (measured ICC ~= 0.61). Pooling molecules would treat ~90 correlated
    observations as independent.
    """
    d_pm, on_pm, off_pm = [], [], []
    win_p, ratio_p = [], []
    n_mol = n_clip = 0

    for i in sorted(per_pocket):
        p = paired(per_pocket[i], mode)
        if not p:
            continue
        on_raw = np.array([a for a, _ in p])
        off_raw = np.array([b for _, b in p])
        if clip is None:
            on, off = on_raw, off_raw
        else:
            on = np.clip(on_raw, -clip, clip)
            off = np.clip(off_raw, -clip, clip)
            n_clip += int(((np.abs(on_raw) > clip) | (np.abs(off_raw) > clip)).sum())
        d = off - on                                   # SBE sign
        d_raw = off_raw - on_raw

        d_pm.append(float(d.mean()))
        on_pm.append(float(on.mean()))
        off_pm.append(float(off.mean()))
        win_p.append(100.0 * float((on_raw < off_raw).mean()))   # rank-based: always raw
        n_mol += len(p)

        if ref is not None and i in ref:
            ref_on, ref_delta = ref[i]                 # rank-based: always raw
            hits = int(((on_raw < ref_on) & (d_raw > ref_delta)).sum())
            ratio_p.append(100.0 * hits / len(p))

    if not d_pm:
        return None
    return {
        'n_pk': len(d_pm),
        'n_mol': n_mol,
        'delta_p_avg': float(np.mean(d_pm)),
        'delta_p_med': float(np.median(d_pm)),
        'on_avg': float(np.mean(on_pm)),
        'on_med': float(np.median(on_pm)),
        'off_avg': float(np.mean(off_pm)),
        'off_med': float(np.median(off_pm)),
        'ratio_avg': float(np.mean(ratio_p)) if ratio_p else None,
        'ratio_med': float(np.median(ratio_p)) if ratio_p else None,
        'win_avg': float(np.mean(win_p)),
        'win_med': float(np.median(win_p)),
        'n_clipped': n_clip,
    }


def restrict(per_pocket, keep):
    return {i: v for i, v in per_pocket.items() if i in keep}


def reference_baseline(data, mode='dock'):
    """{pocket_idx: (on, delta_sbe)} for the crystal ligand, used by the Ratio metric."""
    out = {}
    for i, mols in (data.get('reference') or {}).items():
        p = paired(mols, mode)
        if p:
            on, off = p[0]                             # exactly one crystal ligand per pocket
            out[i] = (on, off - on)                    # (absolute on-target, SBE-sign delta)
    return out



def section(title, lines):
    return ['-' * W, title, '-' * W, ''] + lines + ['']


def table_rows(cfg, data, mode, keep=None, ref=None, clip=CLIP, only=None):
    """`keep` restricts the POCKETS, `only` the ROWS.

    They go together: a pocket set derived from one roster is only meaningful for that
    roster, because a model outside it may not cover every pocket in the set.
    """
    rows, raw = [], {}
    for model in cfg['models']:
        tag = model['tag']
        if only is not None and tag not in only:
            continue
        pp = data.get(tag) or {}
        if keep is not None:
            pp = restrict(pp, keep)
        # Reference cannot beat itself, so Ratio% is undefined for that row, not zero.
        s = stats(pp, mode, ref=None if tag == 'reference' else ref, clip=clip)
        raw[tag] = s
        if s is None:
            rows.append([model['label']] + [''] * (len(HEADERS) - 1))
            continue
        rows.append([
            model['label'], s['n_pk'], s['n_mol'],
            fmt(s['delta_p_avg']), fmt(s['delta_p_med']),
            fmt(s['on_avg']), fmt(s['on_med']),
            fmt(s['off_avg']), fmt(s['off_med']),
            '' if s['ratio_avg'] is None else f"{s['ratio_avg']:.1f}",
            '' if s['ratio_med'] is None else f"{s['ratio_med']:.1f}",
            f"{s['win_avg']:.1f}", f"{s['win_med']:.1f}",
        ])
    return rows, raw


HEADERS = ['Model', 'n_pk', 'n_mol',
           'Delta_P(Avg)', 'Delta_P(Med)',
           'On(Avg)', 'On(Med)', 'Off(Avg)', 'Off(Med)',
           'Ratio%(Avg)', 'Ratio%(Med)', 'Win%(Avg)', 'Win%(Med)']
ALIGNS = ['l'] + ['r'] * (len(HEADERS) - 1)


def sweep_table(cfg, data, mode, keep=None, thresholds=(None, 100, 50, 25, 20, 15)):
    """Delta_P(Avg) at several winsorising thresholds, so the reader can see that the
    reported number does not hinge on the +/-20 choice."""
    rows = []
    for model in cfg['models']:
        pp = data.get(model['tag']) or {}
        if keep is not None:
            pp = restrict(pp, keep)
        cells, n_clipped = [], ''
        for t in thresholds:
            st = stats(pp, mode, clip=t)
            if st is None:
                cells.append('')
                continue
            cells.append(fmt(st['delta_p_avg']))
            if t == CLIP:
                n_clipped = f"{st['n_clipped']} ({100.0 * st['n_clipped'] / st['n_mol']:.2f}%)"
        if cells:
            rows.append([model['label']] + cells + [n_clipped])
    return rows


SWEEP_HEADERS = ['Model', 'raw', '+/-100', '+/-50', '+/-25', '+/-20', '+/-15', 'clipped at +/-20']
SWEEP_ALIGNS = ['l'] + ['r'] * (len(SWEEP_HEADERS) - 1)


def _row_id(delta_tag, registry='configs/models.json'):
    """Registry id for a delta tag. `id` is the join key for the integrated table; the tag is
    family-local and the label is not a key at all."""
    import sys as _s, os as _o
    _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
    import model_registry                                                # noqa: E402
    try:
        return model_registry.resolve('delta', delta_tag, path=registry)['id']
    except KeyError:
        return ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--registry', default=REGISTRY)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--out_root', default='results/offtarget')
    ap.add_argument('--out_dir', default='results/comparison/f4_delta_score')
    args = ap.parse_args()

    cfg = load_registry(args.registry)
    seed = args.seed if args.seed is not None else cfg['seed']
    asg = load_assignment(seed, args.registry)
    tp, entries, idx2full, full2idx = load_test_pockets()
    pocket_name = {e['pocket_idx']: e['pocket'] for e in entries}
    off_of = {a['pocket_idx']: a['off_idx'] for a in asg['assignment']}

    data, missing = collect(cfg, asg, args.out_root)
    strict, _ = collect(cfg, asg, args.out_root, strict_box=True)

    for tag, miss in missing.items():
        if miss:
            print(f'[warn] {tag}: {len(miss)} pockets not docked yet: '
                  f"{miss if len(miss) <= 12 else str(miss[:12]) + ' ...'}")

    covered = {tag: set(pp) for tag, pp in data.items()}
    common = sorted(set.intersection(*covered.values())) if all(covered.values()) else []

    # `common` intersects over EVERY delta model, which is the right scope for the report
    # below but the wrong one for the manuscript: AR and DecompDiff are in no manuscript
    # table, yet their gaps at pockets 25/58/82/83 cost every reported row four pockets.
    # `roster` intersects over the reported models only. Nothing is re-docked -- each model
    # was docked on all 100 pockets and the restriction is applied here, at aggregation.
    roster_tags = [t for t in (delta_tag_of(args.registry, i) for i in report_roster(args.registry))
                   if t in covered]
    roster = (sorted(set.intersection(*(covered[t] for t in roster_tags)))
              if roster_tags and all(covered[t] for t in roster_tags) else [])
    if roster_tags:
        print(f'[roster] {len(roster_tags)} reported models -> {len(roster)} shared pockets '
              f'(all-model common: {len(common)})')
    expanded = {}
    for tag, pp in data.items():
        tot = sum(len(v) for v in pp.values())
        exp = sum(1 for v in pp.values() for m in v if m.get('box_expanded'))
        expanded[tag] = (exp, tot, 100.0 * exp / tot if tot else 0.0)

    os.makedirs(args.out_dir, exist_ok=True)
    lines = ['=' * W,
             'DELTA SCORE (SELECTIVITY) - own pocket vs one random other pocket',
             '=' * W, '']
    lines += [
        'Delta = Vina(off-target) - Vina(on-target), kcal/mol.  HIGHER = MORE SELECTIVE.',
        'Every Delta column uses this SBE-Diff Eq.6 sign (analysis/SBE_Diff_수정.pdf), so the',
        'numbers are directly comparable to their Table 2 / Table 4.',
        '',
        'SIGN TRAP: the sentence under SBE-Diff Eq.6 claims "a smaller value of Delta Score',
        'indicates ... high affinity towards target y_i itself". That contradicts their own',
        'equation AND their own tables (Table 2 heads the column "Delta up"; Reference scores',
        '1.158, the best row). Their Table 3 settles it arithmetically: TargetDiff ori 6.665,',
        'shuffle 6.320, delta 0.335 = 6.665 - 6.320. HIGHER IS BETTER.',
        '',
        'ONE AGGREGATION RULE for every column, in two stages:',
        '',
        '  stage 1  reduce each pocket to ONE number',
        '             Delta / On / Off  ->  that pocket MEAN over its molecules',
        '             Ratio% / Win%     ->  that pocket rate (already one scalar)',
        '  stage 2  combine across pockets',
        '             (Avg) -> mean over pockets      (Med) -> median over pockets',
        '',
        'This is exactly SBE-Diff Table 2: their "Delta mean" is (Avg), their "Delta median"',
        'is (Med) -- and here the same rule applies to On/Off/Ratio%/Win% as well.',
        '',
        'Why a MEAN inside the pocket: SBE-Diff Eq.6 is literally 1/m_i * sum_j, the',
        'unbiasedness argument that justifies n~=1 (their Appendix A.2) needs a mean, and a',
        'within-pocket median would discard the genuinely selective upper tail -- 16.3% of',
        'our molecules sit at delta > +5 kcal/mol after clipping, versus 1.8% for AR.',
        'That tail is signal, and it is what a selectivity metric should reward.',
        '',
        'Why the pocket is the unit: with n~ = 1 every molecule in a pocket faces the SAME',
        'off-target protein, so molecules within a pocket are correlated by construction',
        '(measured ICC ~ 0.61). Pooling molecules would treat ~90 correlated observations as',
        'independent and let a pocket with 100 molecules outweigh one with 9 by 11x.',
        '',
        '(Avg) is the unbiased estimator and every target counts once; (Med) is the',
        'robustness companion -- under repeated pocket resampling it moves 20-35% less than',
        'the mean, which is the n~=1 draw-luck this design is exposed to. Read them together.',
        '',
        '  Ratio%  share of a pocket\'s molecules beating that pocket\'s REFERENCE ligand on BOTH',
        '          the absolute on-target score AND the delta score (SBE-Diff §5.1).',
        '          Blank for the Reference row: it cannot beat itself.',
        '  Win%    share of molecules binding their OWN pocket better than the off-target one.',
        '          Purely rank-based, so grid-penalty blow-ups cannot move it.',
        '',
        f'Off-target: ONE pocket per target, k != i, drawn once with seed {seed} and SHARED BY',
        f'EVERY MODEL ROW (configs/offtarget_assignment_seed{seed}.json). n~ = 1, as in both papers.',
        'The generated ligand is rigidly translated so its mass-weighted centre of mass sits on',
        'the off-target crystal reference ligand centre; box centre = that centre; box size =',
        'ceil(ref_extent/10)*10 + 10 per axis; exhaustiveness 8.',
        '',
        'ON-TARGET NUMBERS ARE REUSED from each model .pt and were measured with a DIFFERENT',
        'protocol: own-bbox box (ligand extent + 5 A) at exhaustiveness 16, around the model\'s',
        'OWN generated pose. That asymmetry is NOT uniform across rows -- it scales with how far',
        'a molecule was tuned to the Vina objective -- and it is why Delta here correlates',
        'r = -0.93 with the absolute on-target score (SBE-Diff\'s Glide table: +0.37).',
        'See analysis/RESULTS_delta_score_2026-08.md sections 2-4 before quoting any ranking.',
        '',
        'Molecules: EVERY docking-successful molecule of every model (no subsampling).',
        'A molecule enters a mode\'s statistics only if it has BOTH an on- and an off-target',
        'value for that mode, so each mode carries its own n.',
        '',
    ]

    for mode in MODES:
        rows, _ = table_rows(cfg, data, mode, ref=reference_baseline(data, mode))
        body = [f'Mode: {MODE_LABEL[mode]}  (Delta / On / Off winsorised at +/-{CLIP:.0f} kcal/mol;',
                f' Win% and Ratio% from raw values)', ''] + render_table(HEADERS, rows, ALIGNS)
        if mode != 'dock':
            body += ['',
                     'A rigidly translated ligand that pokes outside the grid makes Vina return',
                     '~1e6 kcal/mol, which this mode hits often. Read the (Med) columns and Win%;',
                     'the (Avg) columns can be dragged by a single molecule. See the winsorised',
                     'table at the end for how large that effect is.']
        lines += section(f'ALL POCKETS - {MODE_LABEL[mode]}', body)

    if common:
        rows, _ = table_rows(cfg, data, 'dock', keep=set(common),
                             ref=reference_baseline(data, 'dock'))
        lines += section(
            f'COMMON POCKET SET ({len(common)} pockets) - Vina Dock',
            ['Every row restricted to the pockets ALL models cover, so the rows are strictly',
             'comparable. This is the number to quote.', ''] +
            render_table(HEADERS, rows, ALIGNS))

    if roster:
        rows, _ = table_rows(cfg, data, 'dock', keep=set(roster),
                             ref=reference_baseline(data, 'dock'),
                             only=set(roster_tags))
        lines += section(
            f'REPORTED ROSTER ({len(roster)} pockets) - Vina Dock',
            ['The manuscript table. Restricted to the pockets the NINE reported models share,',
             'and to those models: letting a model that appears in no table shrink the pocket',
             'set penalises every row that is shown. Rows here are strictly comparable to each',
             'other, NOT to the all-model section above, which is on a different pocket set.', ''] +
            render_table(HEADERS, rows, ALIGNS))

    rows, _ = table_rows(cfg, strict, 'dock', keep=set(common) if common else None,
                         ref=reference_baseline(strict, 'dock'))
    lines += section(
        'SENSITIVITY: STRICT BOX ONLY - Vina Dock',
        ['Molecules whose ligand did not fit the frozen reference-ligand box were re-docked',
         'once in a box widened just enough to contain them; a wider box can only find MORE',
         'binding modes, i.e. a better off-target score and a LESS negative Delta, so the',
         'fallback is conservative against the selectivity claim. This table drops those',
         'molecules entirely. Box-expansion rate per model:',
         ''] +
        [f'    {m["label"]:26s} {expanded[m["tag"]][0]:4d} / {expanded[m["tag"]][1]:5d} '
         f'= {expanded[m["tag"]][2]:5.2f}%' for m in cfg['models']] +
        ['', 'Restricted to the common pocket set.' if common else ''] +
        render_table(HEADERS, rows, ALIGNS))

    raw_rows, _ = table_rows(cfg, data, 'dock', keep=set(common) if common else None,
                            ref=reference_baseline(data, 'dock'), clip=None)
    lines += section(
        'SIDE: NO CLIPPING (raw values) - Vina Dock',
        ['The same table with NO winsorising, for reference. Every value-based column here is',
         'exposed to the Vina grid artefacts described at the top: a ligand whose atoms fall',
         'outside the box is scored as if that penalty were a binding energy (up to +4813',
         'kcal/mol off-target, +1559 on-target in this dataset). Win% and Ratio% are identical',
         'to the main table -- they are rank-based and were never clipped.',
         'Restricted to the common pocket set.' if common else '',
         ''] +
        render_table(HEADERS, raw_rows, ALIGNS) +
        ['',
         'Threshold sweep on Delta_P(Avg) -- why +/-20 is not a load-bearing choice:',
         ''] +
        render_table(SWEEP_HEADERS,
                     sweep_table(cfg, data, 'dock', set(common) if common else None),
                     SWEEP_ALIGNS) +
        ['',
         'Between +/-100 and +/-20 every model agrees to within ~0.05 kcal/mol. The large move',
         'is raw -> any clipping (AliDiff 2.371 -> ~1.44). Below about +/-15 the clip starts',
         'cutting real values, visible as the Reference row finally moving. So the reported',
         'numbers are insensitive to the threshold anywhere in the physically meaningless',
         'range, which is the property that makes the choice defensible.'])

    lines += section(
        'CAVEATS',
        ['1. FRAGMENTATION MNAR. Eligibility required an on-target Vina Dock score, so molecules',
         '   that fragmented and never docked are absent by construction. Our models fragment',
         '   15-28%; every external baseline .pt stored docking-successful molecules only, so',
         '   their denominator is their full generated set. Delta is a WITHIN-MOLECULE difference,',
         '   which cancels much of the level shift a survivor population causes -- but it does not',
         '   cancel selection on SHAPE, and shape drives off-target fit. Quote the fragmentation',
         '   rate from analysis/RESULTS_pose_fidelity_2026-07.md next to these numbers.',
         '2. n~ = 1. One off-target pocket, one seed. Same as the Delta Score paper, but per-pocket',
         '   variance is large: do NOT interpret an individual pocket Delta, only model-level',
         '   aggregates.',
         '3. Reference (native) contributes exactly 1 molecule per pocket. Its n is ~100 against',
         '   ~9000 for the model rows, so its interval is far wider, and several columns go',
         '   DEGENERATE for that row: with one molecule a pocket mean equals its median, so',
         '   Delta_P(Avg) == Delta_P(Med) == Delta_All(Avg) by construction, and the per-pocket',
         '   Win% can only be 0 or 100, which makes Win%(Med) exactly 100.0 and meaningless.',
         '   Read Delta_All(Med) and Win%(Avg) for the Reference row; ignore its Win%(Med).',
         '3b. Off(Avg) vs Off(Med) diverging by more than ~0.5 flags outlier contamination in',
         '   that row (AliDiff: -6.47 vs -7.60, from off-target values reaching -4823 kcal/mol).',
         '   Prefer the (Med) columns wherever the two disagree.',
         '4. Unequal pocket coverage (AR 97, DecompDiff 99, Ours-ON 99). The COMMON POCKET SET',
         '   table above is the comparable one; the all-pockets table averages over different',
         '   pocket sets per row.'])

    txt_path = os.path.join(args.out_dir, 'delta_score_tables.txt')
    with open(txt_path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')

    # ---- CSVs ------------------------------------------------------------- #
    overall = os.path.join(args.out_dir, 'delta_score_overall.csv')
    with open(overall, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['id', 'model', 'label', 'mode', 'arm', 'n_pk', 'n_mol',
                    'delta_p_avg', 'delta_p_med',
                    'on_avg', 'on_med', 'off_avg', 'off_med',
                    'ratio_pct_avg', 'ratio_pct_med', 'win_pct_avg', 'win_pct_med',
                    'n_clipped'])
        # `only` goes with `keep`: a pocket set derived from one roster is meaningful only
        # for that roster. AR covers 97 pockets and DecompDiff 99, so scoring them on the
        # 9-model set would put them on a different pocket set from the rows beside them --
        # the very thing this arm exists to prevent.
        for arm, dset, keep, only in (('all', data, None, None),
                                      ('common', data, set(common) if common else None, None),
                                      ('roster', data, set(roster) if roster else None,
                                       set(roster_tags) if roster else set()),
                                      ('strict_box', strict, set(common) if common else None, None)):
            for mode in MODES:
                ref = reference_baseline(dset, mode)
                for model in cfg['models']:
                    if only is not None and model['tag'] not in only:
                        continue
                    pp = dset.get(model['tag']) or {}
                    if keep is not None:
                        pp = restrict(pp, keep)
                    s_ = stats(pp, mode, ref=None if model['tag'] == 'reference' else ref)
                    if s_ is None:
                        continue
                    g = lambda k, nd=4: '' if s_[k] is None else f'{s_[k]:.{nd}f}'
                    w.writerow([_row_id(model['tag']), model['tag'], model['label'], mode, arm, s_['n_pk'], s_['n_mol'],
                                g('delta_p_avg'), g('delta_p_med'),
                                g('on_avg'), g('on_med'), g('off_avg'), g('off_med'),
                                g('ratio_avg', 2), g('ratio_med', 2),
                                g('win_avg', 2), g('win_med', 2), s_['n_clipped']])

    per_target = os.path.join(args.out_dir, 'delta_score_per_target.csv')
    with open(per_target, 'w', newline='') as fh:
        w = csv.writer(fh)
        # delta = off - on (SBE sign, higher = more selective), one row per (pocket, model, mode)
        w.writerow(['pocket_idx', 'pocket', 'off_pocket_idx', 'off_pocket', 'model', 'mode',
                    'n_mol', 'delta_mean', 'delta_med', 'on_mean', 'on_med', 'off_mean',
                    'off_med', 'win_pct'])
        for model in cfg['models']:
            for i, mols in sorted((data.get(model['tag']) or {}).items()):
                for mode in MODES:
                    p = paired(mols, mode)
                    if not p:
                        continue
                    onv = np.asarray([on for on, _ in p]); offv = np.asarray([off for _, off in p])
                    d = offv - onv
                    w.writerow([i, pocket_name[i], off_of[i], pocket_name[off_of[i]],
                                model['tag'], mode, len(p), f'{d.mean():.4f}',
                                f'{np.median(d):.4f}',
                                f'{onv.mean():.4f}', f'{np.median(onv):.4f}',
                                f'{offv.mean():.4f}', f'{np.median(offv):.4f}',
                                f'{100 * np.mean(onv < offv):.2f}'])

    per_mol = os.path.join(args.out_dir, 'delta_score_per_mol.csv')
    with open(per_mol, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['model', 'pocket_idx', 'mol_index', 'off_pocket_idx', 'smiles',
                    'on_score', 'on_min', 'on_dock', 'off_score', 'off_min', 'off_dock',
                    'delta_dock_sbe', 'box_expanded'])
        for model in cfg['models']:
            for i, mols in sorted((data.get(model['tag']) or {}).items()):
                for m in mols:
                    on, off = m['on'], m['off']
                    dd = (None if on.get('dock') is None or off.get('dock') is None
                          else off['dock'] - on['dock'])       # SBE sign: higher = selective
                    w.writerow([model['tag'], i, m['mol_index'], m['off_pocket_idx'],
                                m.get('smiles', ''),
                                fmt(on.get('score_only')), fmt(on.get('minimize')),
                                fmt(on.get('dock')), fmt(off.get('score_only')),
                                fmt(off.get('minimize')), fmt(off.get('dock')),
                                fmt(dd), int(bool(m.get('box_expanded')))])

    print(f'wrote {txt_path}')
    print(f'wrote {overall}')
    print(f'wrote {per_target}')
    print(f'wrote {per_mol}')
    print(f'\ncommon pocket set: {len(common)} pockets')
    for model in cfg['models']:
        s_ = stats(restrict(data.get(model['tag']) or {}, set(common)) if common
                   else (data.get(model['tag']) or {}), 'dock')
        if s_:
            print(f"  {model['label']:26s} Delta_P(Avg) {s_['delta_p_avg']:7.3f}  "
                  f"Delta_P(Med) {s_['delta_p_med']:7.3f}  "
                  f"Off(Med) {s_['off_med']:7.3f}  "
                  f"Win%(Avg) {s_['win_avg']:5.1f}  n={s_['n_mol']}")


if __name__ == '__main__':
    main()
