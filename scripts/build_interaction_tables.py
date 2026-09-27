"""Render the PLIP interaction analysis as a single readable report.

Same spirit as scripts/build_comparison_tables.py: one plain-text file, section per question,
so the whole picture can be read without opening a notebook.

Sections
  1  분자당 상호작용 개수         절대량. native와 무관.
  2  native 대비 재현 정확도       IFP Tanimoto / recall / precision / F1.
  3  recall vs precision 진단     recall 우위가 배치인지 물량인지 가른다.
  4  가이던스 ON/OFF              같은 체크포인트, 가이던스만 차이.
  5  교락 통제                     크기 매칭 + 조성.
  6  특이성 음성대조                포켓 뒤섞기.
  7  포켓 대응 유의성               paired Wilcoxon.

Usage:
    python scripts/build_interaction_tables.py --dir eval_plip --out results/comparison/appendix/ifp/interaction_tables.txt
    python scripts/build_interaction_tables.py --docked_only ...     # comparable population
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import stats as st

W = 160
MIN_REF_HEAVY = 12

# Set once in main() from the records actually loaded: True only when EVERY interaction entry
# carries the protein-atom index `pa` (eval_plip_atom/), which the atom-level IFP needs.
HAS_PA = False

# display order; (tag, label, group)
ORDER = [
    ('NATIVE',                    'NATIVE (crystal ligand)',      'ref'),
    ('vina_fixed_best',           'vina_fixed_best',              'ours'),
    ('pignet_fixed_best',         'pignet_fixed_best',            'ours'),
    ('pidiff_exact',              'pidiff_exact (vdW only)',      'ours'),
    # physics-free control: identical to the vina config except loss_vdw_weight = 0
    ('novdw',                     'novdw (no physics loss)',      'abl'),
    ('targetdiff',                'targetdiff (external)',        'ext'),
    ('kgdiff (external)',         'kgdiff (external)',            'ext'),
    ('kgdiff',                    'kgdiff (external)',            'ext'),
    ('ipdiff',                    'IPDiff (external, docked in-house)', 'ext'),
    ('vina_fixed_best_typeonly',  'vina_fixed_best [TYPE only]',  'abl'),
    ('vina_fixed_best_posonly',   'vina_fixed_best [POS only]',   'abl'),
    ('vina_fixed_best_noguide',   'vina_fixed_best  [guide OFF]', 'ng'),
    ('pignet_fixed_best_noguide', 'pignet_fixed_best [guide OFF]', 'ng'),
    ('vina_noguide',              'vina_675k        [guide OFF]', 'ng25'),
    ('pignet_noguide',            'pignet_675k      [guide OFF]', 'ng25'),
    ('pidiff_noguide',            'pidiff_342k      [guide OFF]', 'ng25'),
    ('pgdiff_noguide',            'pgdiff_342k      [guide OFF]', 'ng25'),
    # Pose-profile rows: the REDOCKED / MINIMIZED pose of our own model, not separate models.
    # Both carry a DOCKED_SOURCE entry of kind 'none' because a redocked pose only exists for a
    # molecule that docked (verified key-for-key against eval_out/vina_fixed/docked_names.txt),
    # so they satisfy the docking-successful convention by construction. They were reaching the
    # table only through the auto-discovery fallback removed in main(); listed explicitly here so
    # they survive that removal. Group 'other' reproduces their previous placement exactly.
    ('vina_fixed_best_dock',      'vina_fixed_best_dock',         'other'),
    ('vina_fixed_best_minimize',  'vina_fixed_best_minimize',     'other'),
]


def hr(c='-'):
    return c * W


def head(title, c='='):
    return f'{hr(c)}\n{title}\n{hr(c)}'


# --------------------------------------------------------------------- docking-successful filter
# Analysis convention (see scripts/mark_docked.py): every table is reported on DOCKING-SUCCESSFUL
# molecules only, because the reference .pt files were saved that way. PLIP was run on the FULL
# generated set, so the restriction has to happen here.
#
# The join key is `p{pocket_idx:03d}_m{mi:04d}`, the same string mark_docked.py writes. It is only
# valid for tags PLIP read through `--pt` (load_grouped), where `mi` is the molecule's index in the
# pocket list. Tags PLIP read through `--samples` use load_samples, which SKIPS unreconstructible
# and fragmented molecules before indexing, so `mi` does not line up with the .pt at all — and by
# the docked<=>connected equivalence those rows are already the docking-successful set. They are
# listed as 'samples' and left untouched.
DOCKED_SOURCE = {
    'targetdiff':                ('pt', 'results/sampling_results/targetdiff_vina_docked.pt'),
    'kgdiff':                    ('pt', 'results/sampling_results/our_vina_score_docked.pt'),
    'pidiff_exact':              ('pt', 'results/head1_dock_342k/head1_dock_342k_vina_docked.pt'),
    'vina_fixed_best':           ('pt', 'results/vina_fixed_best_gen/'
                                        'vina_fixed_best_vina_docked.pt'),
    'pignet_fixed_best':         ('pt', 'results/pignet_fixed_best_gen/'
                                        'pignet_fixed_best_vina_docked.pt'),
    'novdw':                     ('pt', 'results/novdw_gen/novdw_gen_vina_docked.pt'),
    # IPDiff official samples docked in-house (own-bbox, exh 16). MERGED pt (ipdiff_merge_docking.py),
    # NOT consolidate's output: the merged one keeps 100 entries/pocket so its p<NNN>_m<MMMM>
    # positions match eval_plip/ipdiff. See ipdiff-vs-pidiff-sources memo.
    'ipdiff':                    ('pt', 'results/ipdiff_official_gen/'
                                        'ipdiff_official_vina_docked.pt'),
    'vina_fixed_best_typeonly':  ('pt', 'results/typeonly/vina_fixed_best_typeonly/'
                                        'vina_fixed_best_typeonly_vina_docked.pt'),
    'vina_fixed_best_posonly':   ('pt', 'results/posonly/vina_fixed_best_posonly/'
                                        'vina_fixed_best_posonly_vina_docked.pt'),
    'vina_fixed_best_noguide':   ('samples', None),
    'pignet_fixed_best_noguide': ('samples', None),
    'vina_noguide':              ('samples', None),
    'pignet_noguide':            ('samples', None),
    'pidiff_noguide':            ('samples', None),
    'pgdiff_noguide':            ('samples', None),
    'NATIVE':                    ('none', None),      # crystal ligands; nothing to restrict
    # scripts/plip_pose.py profiles the REDOCKED / MINIMIZED pose, which only exists for a
    # molecule that docked — verified key-for-key equal to eval_out/vina_fixed/docked_names.txt.
    'vina_fixed_best_dock':      ('none', None),
    'vina_fixed_best_minimize':  ('none', None),
}


def _has_aff(vina, key):
    """mark_docked.py:has_aff — a valid AutoDock Vina result for that mode."""
    e = (vina or {}).get(key)
    if isinstance(e, list) and e:
        e = e[0]
    return isinstance(e, dict) and e.get('affinity') is not None


def docked_keys(pt_path):
    """-> set of 'pNNN_mMMMM' for the docking-successful molecules of a grouped .pt.

    Uses plip_interactions.load_grouped so the (pocket, molecule) indexing is byte-for-byte the
    one that produced the JSONL — deriving it any other way risks a silent off-by-one.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from plip_interactions import load_grouped                          # noqa: E402

    data = load_grouped(pt_path)
    key = 'dock' if any(_has_aff(e.get('vina'), 'dock') for mols in data for e in mols) \
        else 'minimize'
    return {f'p{pi:03d}_m{mi:04d}'
            for pi, mols in enumerate(data)
            for mi, e in enumerate(mols) if _has_aff(e.get('vina'), key)}


def load(d, tag, docked_only=False, log=None):
    rows = []
    for p in sorted(glob.glob(os.path.join(d, tag, 'pocket*.jsonl'))):
        with open(p) as f:
            rows += [json.loads(l) for l in f]
    if not docked_only or not rows:
        return rows
    kind, src = DOCKED_SOURCE.get(tag, (None, None))
    if kind == 'pt' and os.path.isfile(src):
        keep = docked_keys(src)
        before = len(rows)
        rows = [r for r in rows
                if f"p{int(r['pocket_idx']):03d}_m{int(r['mi']):04d}" in keep]
        note = f'{before} -> {len(rows)}'
    elif kind == 'samples':
        note = f'{len(rows)} (already docking-successful: --samples drops fragmented molecules)'
    elif kind == 'none':
        note = f'{len(rows)} (reference)'
    else:
        note = f'{len(rows)} *** NO SOURCE MAPPED — still the full generated set ***'
    if log is not None:
        log.append((tag, note))
    return rows


def ifp(row, kinds=('hbond', 'hydrophobic'), level='residue'):
    """Interaction-fingerprint bits.

    level='residue' (historical default for every table here):
        bit = (chain, resnr, restype, kind). Three contacts to the SAME residue collapse into one
        bit, so a generated molecule gets credit for "hitting the right residue".
    level='atom':
        bit = (chain, resnr, restype, PROTEIN ATOM index, kind). Credit only for hitting the SAME
        protein atom. This is DiffInt's definition (JCIM 2025, 65, 71-82, "reconstruction of
        hydrogen bonds ... bonds with the correct protein atoms"), and is strictly STRICTER --
        atom-level recall/F1 can never exceed the residue-level value on the same data.
        Requires the `pa` field, present only in eval_plip_atom/ (scripts/sbatch_plip_atom.sh).
    """
    if level == 'atom':
        out = set()
        for k in kinds:
            for x in row.get(k, []):
                if 'pa' not in x:
                    raise KeyError("atom-level IFP needs the 'pa' field; re-run PLIP via "
                                   "scripts/sbatch_plip_atom.sh, or use level='residue'")
                out.add((x['chain'], x['resnr'], x['restype'], x['pa'], k))
        return out
    return {(x['chain'], x['resnr'], x['restype'], k)
            for k in kinds for x in row.get(k, [])}


def tanimoto(a, b):
    if not a and not b:
        return np.nan
    u = len(a | b)
    return len(a & b) / u if u else np.nan


def prf(g, r):
    if not r:
        return np.nan, np.nan, np.nan
    rec = len(g & r) / len(r)
    pre = len(g & r) / len(g) if g else np.nan
    f1 = np.nan if (np.isnan(pre) or pre + rec == 0) else 2 * pre * rec / (pre + rec)
    return pre, rec, f1


def pocket_stats(rows, ref, size=None):
    if size:
        rows = [r for r in rows if size[0] <= r['heavy'] <= size[1]]
    if not rows:
        return None
    RA, RH, RY = ifp(ref), ifp(ref, ('hbond',)), ifp(ref, ('hydrophobic',))
    # atom-level reference bits, when the PLIP run recorded protein atom indices (`pa`).
    # Same maths, stricter bits -> the *_at metrics are the DiffInt-comparable ones.
    #
    # Deciding this from the reference ALONE is not safe: a pocket whose native ligand recorded no
    # interactions at all makes ifp(ref, level='atom') return an empty set without ever touching
    # `pa`, atom_ok comes back True, and the first model row that DOES have interactions then
    # raises KeyError and kills the whole run (this is why eval_plip/ could not be re-rendered).
    # `has_pa` is therefore decided once, in main(), over every record actually loaded.
    atom_ok = HAS_PA
    if atom_ok:
        RA_a = ifp(ref, level='atom')
        RH_a = ifp(ref, ('hbond',), level='atom')
        RY_a = ifp(ref, ('hydrophobic',), level='atom')
    acc = defaultdict(list)
    for r in rows:
        acc['hb'].append(r['n_hbond'])
        acc['hy'].append(r['n_hydrophobic'])
        acc['heavy'].append(r['heavy'])
        acc['logp'].append(r['logp'])
        acc['pct_no'].append(r['pct_no'])
        acc['hb_h'].append(r['n_hbond'] / r['heavy'] if r['heavy'] else np.nan)
        acc['hy_h'].append(r['n_hydrophobic'] / r['heavy'] if r['heavy'] else np.nan)
        acc['p_hb'].append(float(r['n_hbond'] > 0))
        acc['p_hy'].append(float(r['n_hydrophobic'] > 0))
        acc['tani'].append(tanimoto(ifp(r), RA))
        p, q, f = prf(ifp(r, ('hbond',)), RH)
        acc['pre_hb'].append(p); acc['rec_hb'].append(q); acc['f1_hb'].append(f)
        p, q, f = prf(ifp(r, ('hydrophobic',)), RY)
        acc['pre_hy'].append(p); acc['rec_hy'].append(q); acc['f1_hy'].append(f)
        if atom_ok:
            acc['tani_at'].append(tanimoto(ifp(r, level='atom'), RA_a))
            p, q, f = prf(ifp(r, ('hbond',), level='atom'), RH_a)
            acc['pre_hb_at'].append(p); acc['rec_hb_at'].append(q); acc['f1_hb_at'].append(f)
            p, q, f = prf(ifp(r, ('hydrophobic',), level='atom'), RY_a)
            acc['pre_hy_at'].append(p); acc['rec_hy_at'].append(q); acc['f1_hy_at'].append(f)
    out = {k: float(np.nanmean(v)) if len(v) else np.nan for k, v in acc.items()}
    out['n_mol'] = len(rows)
    return out


def paired(a, b):
    pr = [(x, y) for x, y in zip(a, b) if not (np.isnan(x) or np.isnan(y))]
    if len(pr) < 5:
        return len(pr), np.nan, np.nan
    x = np.array([p[0] for p in pr]); y = np.array([p[1] for p in pr])
    if np.allclose(x, y):
        return len(pr), 0.0, 1.0
    return len(pr), float(np.median(x - y)), float(st.wilcoxon(x, y).pvalue)


def sig(p):
    if np.isnan(p):
        return '  '
    return '***' if p < 1e-3 else ' **' if p < 1e-2 else '  *' if p < 0.05 else 'n.s'


def fmt(v, w=8, d=3):
    return f'{"-":>{w}}' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{v:{w}.{d}f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='eval_plip')
    ap.add_argument('--ref', default='NATIVE')
    ap.add_argument('--focus', default='vina_fixed_best')
    ap.add_argument('--out', default='results/comparison/appendix/ifp/interaction_tables.txt')
    ap.add_argument('--size', nargs=2, type=int, default=[20, 25])
    # D5: the docking-successful population is NOT optional. It used to be an opt-in flag, so
    # the same command could emit two incomparable versions of this table -- and both shipped
    # (interaction_tables.txt and interaction_tables_unfiltered.txt). External papers ship .pt
    # files already filtered this way, so including our failures would evaluate us on a wider
    # population. --unfiltered still exists for diagnosis but may only write outside the
    # comparison tree.
    ap.add_argument('--unfiltered', action='store_true',
                    help='DIAGNOSTIC ONLY: skip the docking-successful filter. Refuses to write '
                         'into results/comparison/ -- use results/diagnostics/.')
    a = ap.parse_args()
    a.docked_only = not a.unfiltered
    if a.unfiltered and 'results/comparison' in os.path.abspath(a.out):
        raise SystemExit('--unfiltered may not write into results/comparison/ -- '
                         'point --out at results/diagnostics/')

    present = {os.path.basename(p) for p in glob.glob(os.path.join(a.dir, '*')) if os.path.isdir(p)}
    seen, order = set(), []
    for t, lab, grp in ORDER:
        if t in present and t not in seen:
            seen.add(t); order.append((t, lab, grp))

    # Unregistered tags are REPORTED, never added (fixed 2026-08-21).
    # This used to be `order.append((t, t, 'other'))` for every directory under --dir. Any tag
    # reached that way has no DOCKED_SOURCE entry, so load() leaves it on the FULL generated set
    # while every registered row is docking-successful-only -- two populations in one table, with
    # only a per-row note to say so. It was harmless when ORDER covered eval_plip/, and stopped
    # being harmless as the directory grew: today it would silently append 21 rows, including the
    # 13 abl_* term-ablation arms, `pidiff_baseline` (the PUBLISHED PIDiff, which would sit
    # unfiltered right next to our own `pidiff_exact`), and vina_fixed_best_{dock,minimize}
    # (re-docked poses of our own model, not models at all).
    unregistered = sorted(present - seen - {a.ref})
    if unregistered:
        print(f'[roster] {len(unregistered)} tag(s) under {a.dir}/ are NOT in ORDER and were '
              f'SKIPPED (they have no DOCKED_SOURCE, so including them would mix populations):')
        for t in unregistered:
            print(f'[roster]   - {t}')
        print('[roster] To report one, add it to ORDER *and* to DOCKED_SOURCE.')

    filt = []
    raw = {t: load(a.dir, t, a.docked_only, filt) for t, _, _ in order}
    refs = {r['pk']: r for r in load(a.dir, a.ref, a.docked_only, filt)}
    drug = {k for k, r in refs.items() if r['heavy'] >= MIN_REF_HEAVY}

    # Atom-level bits need `pa` on every entry; decide it from the data, never from the reference
    # alone (see pocket_stats). eval_plip/ has no `pa`, eval_plip_atom/ has it throughout.
    global HAS_PA
    HAS_PA = all('pa' in x
                 for rows in list(raw.values()) + [list(refs.values())]
                 for r in rows for k in ('hbond', 'hydrophobic') for x in r.get(k, []))
    for t, note in filt:
        print(f'[docked_only] {t}: {note}')

    def build(size=None, keys=None):
        out = {}
        for t, _, _ in order:
            by = defaultdict(list)
            for r in raw[t]:
                by[r['pk']].append(r)
            per = {}
            for k, v in by.items():
                if keys is not None and k not in keys:
                    continue
                if k in refs:
                    s = pocket_stats(v, refs[k], size)
                    if s:
                        per[k] = s
            out[t] = per
        return out

    P_all = build()
    P_drug = build(keys=drug)
    P_size = build(size=tuple(a.size), keys=drug)
    L = []

    def mean(P, t, key):
        # .get: the atom-level (*_at) keys are absent when PLIP was run without the `pa` field
        v = [P[t][k].get(key, np.nan) for k in P[t]]
        v = [x for x in v if not np.isnan(x)]
        return float(np.mean(v)) if v else np.nan

    # ---------------------------------------------------------------- header
    L += [head('PLIP INTERACTION ANALYSIS — H-bond / hydrophobic reproduction', '='), '']
    L += ['생성 포즈 그대로 측정 (재도킹 아님). PLIP 3.0.0: H-bond d(D,A)<4.1 A & angle(D-H...A)>100 deg,',
          '소수성 = 자격 탄소쌍 <4.0 A (리간드원자x잔기 단위로 중복제거).',
          'vina 모델은 Vina의 hbonding/hydrophobic 항을 최소화하도록 훈련됐으므로 Vina로 채점하면 순환논증이다.',
          'PLIP은 각도 기준을 쓰는 독립 계측기 — Vina에는 각도 항이 없다.', '']
    if a.docked_only:
        L += ['모집단: 도킹 성공 분자만. 참조 .pt(targetdiff/kgdiff)는 애초에 도킹 성공 분자만 담고 있어서,',
              '필터 없이 비교하면 우리 행만 도킹 실패(=파편화) 분자를 포함한 superset이 된다. 이 필터를 걸면',
              '참조 행은 그대로이고 우리 행만 줄어든다 — 유일하게 모든 행이 같은 것을 세는 설정.',
              'guide OFF 계열은 --samples 경로가 파편화 분자를 먼저 버리므로 이미 도킹 성공 집합이다.', '']
    else:
        L += ['*** 필터 없음: 우리 행은 전체 생성 분자, 참조 행(targetdiff/kgdiff)은 도킹 성공 분자만이다.',
              '    행끼리 직접 비교하지 말 것. --docked_only 로 다시 생성해 비교하라. ***', '']

    # ---------------------------------------------------------------- 1
    L += [head('1. 분자당 상호작용 개수  (절대량, native와 무관, 전체 100포켓)'
               + ('  · 도킹 성공만' if a.docked_only else '  · 필터 없음')), '']
    L += [f'{"model":32s} {"pockets":>7s} {"mols":>6s} | {"H-bond/mol":>11s} {"SD":>5s} {"med":>4s} '
          f'{"P>=1":>6s} | {"hydroph/mol":>12s} {"SD":>5s} {"med":>4s} {"P>=1":>6s} | '
          f'{"HB/heavy":>9s} {"HY/heavy":>9s} {"heavy":>6s}']
    L += [hr()]
    prev = None
    for t, lab, grp in order:
        if grp != prev and prev is not None:
            L.append('')
        prev = grp
        r = raw[t]
        if not r:
            continue
        hb = np.array([x['n_hbond'] for x in r], float)
        hy = np.array([x['n_hydrophobic'] for x in r], float)
        hv = np.array([x['heavy'] for x in r], float)
        npk = len({x['pk'] for x in r})
        L += [f'{lab:32s} {npk:7d} {len(r):6d} | {hb.mean():11.2f} {hb.std():5.2f} {np.median(hb):4.0f} '
              f'{np.mean(hb > 0):6.3f} | {hy.mean():12.2f} {hy.std():5.2f} {np.median(hy):4.0f} '
              f'{np.mean(hy > 0):6.3f} | {np.mean(hb / hv):9.4f} {np.mean(hy / hv):9.4f} {hv.mean():6.2f}']
    L += ['', '  25포켓 세트는 포켓 집합이 달라 100포켓 행과 직접 비교 불가.',
          '  SD가 평균에 육박한다(분자별 편차가 큼). 중앙값 < 평균 = 우편향 분포.', '']

    # ---------------------------------------------------------------- 2
    L += [head(f'2. NATIVE 대비 재현 정확도  (drug-like reference {len(drug)}포켓; native heavy>={MIN_REF_HEAVY})'), '']
    L += ['  IFP 비트 = (chain, resnr, restype, kind). 잔기 단위 이진화 — 같은 잔기와 3번 접촉해도 1비트.',
          '  Tanimoto = |G∩R|/|G∪R| (여분도 벌함)   recall = |G∩R|/|R| (여분 안 벌함)   precision = |G∩R|/|G|', '']
    L += [f'{"model":32s} {"IFP Tani":>9s} | {"HB F1":>7s} {"HB rec":>7s} {"HB prec":>8s} | '
          f'{"HY F1":>7s} {"HY rec":>7s} {"HY prec":>8s}']
    L += [hr()]
    prev = None
    for t, lab, grp in order:
        if not P_drug.get(t):
            continue
        if grp != prev and prev is not None:
            L.append('')
        prev = grp
        L += [f'{lab:32s} {fmt(mean(P_drug, t, "tani"), 9)} | {fmt(mean(P_drug, t, "f1_hb"), 7)} '
              f'{fmt(mean(P_drug, t, "rec_hb"), 7)} {fmt(mean(P_drug, t, "pre_hb"), 8)} | '
              f'{fmt(mean(P_drug, t, "f1_hy"), 7)} {fmt(mean(P_drug, t, "rec_hy"), 7)} '
              f'{fmt(mean(P_drug, t, "pre_hy"), 8)}']
    L += ['']

    # ---------------------------------------------------------------- 2b (atom-level IFP)
    if any(not np.isnan(mean(P_drug, t, 'rec_hb_at')) for t, _, _ in order if P_drug.get(t)):
        L += [head('2b. NATIVE 대비 재현 정확도 — 원자 단위 IFP  (DiffInt 정의)'), '']
        L += ['  IFP 비트 = (chain, resnr, restype, PROTEIN ATOM, kind).',
              '  §2와의 차이: §2는 잔기 단위라 같은 잔기의 어느 원자에 맺히든 1비트로 인정한다.',
              '  여기서는 "올바른 단백질 원자"에 맺혀야 인정 → 항상 §2보다 엄격하고, 같은 데이터에서',
              '  recall/F1이 §2 값을 넘을 수 없다. 두 값의 격차 = 잔기는 맞췄지만 원자는 틀린 접촉의 비율.',
              '',
              '  DiffInt (JCIM 2025, 65, 71-82) 의 "reconstruction of hydrogen bonds"가 이 정의이지만,',
              '  그 논문은 (a) H결합 0개 레퍼런스를 제외한 93포켓, (b) 한 원자에 여러 결합 시 100% 초과 허용,',
              '  (c) 다른 H결합 판정 도구를 쓴다. 따라서 아래 HB rec를 DiffInt 표와 직접 나란히 놓으면 안 된다.', '']
        L += [f'{"model":32s} {"IFP Tani":>9s} | {"HB F1":>7s} {"HB rec":>7s} {"HB prec":>8s} | '
              f'{"HY F1":>7s} {"HY rec":>7s} {"HY prec":>8s} | {"HBrec 잔기":>10s} {"→원자":>7s} {"손실":>7s}']
        L += [hr()]
        prev = None
        for t, lab, grp in order:
            if not P_drug.get(t):
                continue
            if grp != prev and prev is not None:
                L.append('')
            prev = grp
            r_res, r_at = mean(P_drug, t, 'rec_hb'), mean(P_drug, t, 'rec_hb_at')
            drop = (r_at - r_res) if (not np.isnan(r_res) and not np.isnan(r_at)) else np.nan
            L += [f'{lab:32s} {fmt(mean(P_drug, t, "tani_at"), 9)} | {fmt(mean(P_drug, t, "f1_hb_at"), 7)} '
                  f'{fmt(mean(P_drug, t, "rec_hb_at"), 7)} {fmt(mean(P_drug, t, "pre_hb_at"), 8)} | '
                  f'{fmt(mean(P_drug, t, "f1_hy_at"), 7)} {fmt(mean(P_drug, t, "rec_hy_at"), 7)} '
                  f'{fmt(mean(P_drug, t, "pre_hy_at"), 8)} | {fmt(r_res, 10)} {fmt(r_at, 7)} {fmt(drop, 7)}']
        L += ['']

    # ---------------------------------------------------------------- 3
    L += [head('3. 진단: recall 우위는 배치인가 물량인가'), '']
    L += ['  recall은 여분 접촉을 벌하지 않으므로 "많이 쏘면" 오른다. precision과 함께 읽어야 한다.',
          '  recall 순위 >> F1 순위 이면 그 우위는 배치가 아니라 물량에서 온 것.', '']
    L += [f'{"model":32s} {"HY/mol":>7s} {"HY rec 순위":>12s} {"HY F1 순위":>11s} {"판정":>28s}']
    L += [hr()]
    cand = [t for t, _, _ in order if t != a.ref and P_drug.get(t)]
    r_rec = {t: i + 1 for i, t in enumerate(sorted(cand, key=lambda x: -mean(P_drug, x, 'rec_hy')))}
    r_f1 = {t: i + 1 for i, t in enumerate(sorted(cand, key=lambda x: -mean(P_drug, x, 'f1_hy')))}
    for t, lab, _ in order:
        if t not in r_rec:
            continue
        d = r_f1[t] - r_rec[t]
        v = '물량 효과 (recall만 높음)' if d >= 2 else ('배치 우수 (F1이 더 높음)' if d <= -2 else '중립')
        hy = np.mean([x['n_hydrophobic'] for x in raw[t]])
        L += [f'{lab:32s} {hy:7.2f} {r_rec[t]:12d} {r_f1[t]:11d} {v:>28s}']
    L += ['']

    # ---------------------------------------------------------------- 4
    L += [head('4. 가이던스 ON vs OFF  (같은 체크포인트, affinity guidance만 차이)'), '']
    L += [f'{"model":32s} {"guide":>6s} {"HB/mol":>7s} {"HY/mol":>7s} {"IFP Tani":>9s} {"HB F1":>7s} '
          f'{"HY F1":>7s} {"logP":>6s} {"%N+O":>6s} {"HY/heavy":>9s}']
    L += [hr()]
    for on, off, nm in [('vina_fixed_best', 'vina_fixed_best_noguide', 'vina_fixed_best'),
                        ('pignet_fixed_best', 'pignet_fixed_best_noguide', 'pignet_fixed_best')]:
        for t, g in ((on, 'ON'), (off, 'OFF')):
            if not P_drug.get(t):
                continue
            hb = np.mean([x['n_hbond'] for x in raw[t]])
            hy = np.mean([x['n_hydrophobic'] for x in raw[t]])
            L += [f'{nm:32s} {g:>6s} {hb:7.2f} {hy:7.2f} {fmt(mean(P_drug, t, "tani"), 9)} '
                  f'{fmt(mean(P_drug, t, "f1_hb"), 7)} {fmt(mean(P_drug, t, "f1_hy"), 7)} '
                  f'{fmt(mean(P_drug, t, "logp"), 6, 2)} {fmt(mean(P_drug, t, "pct_no"), 6)} '
                  f'{fmt(mean(P_drug, t, "hy_h"), 9, 4)}']
        if P_drug.get(on) and P_drug.get(off):
            ks = sorted(set(P_drug[on]) & set(P_drug[off]))
            L.append(f'{"  -> OFF - ON (paired, n=" + str(len(ks)) + ")":32s} {"":>6s} '
                     + ' '.join(f'{"":>7s}' for _ in range(2)) + ' '
                     + ' '.join(
                         f'{paired([P_drug[off][k][m] for k in ks], [P_drug[on][k][m] for k in ks])[1]:+7.3f}'
                         f'{sig(paired([P_drug[off][k][m] for k in ks], [P_drug[on][k][m] for k in ks])[2])}'
                         for m in ('tani', 'f1_hb', 'f1_hy')))
        L.append('')
    ref_lp = mean(P_drug, a.ref, 'logp') if P_drug.get(a.ref) else np.nan
    L += [f'  NATIVE 기준값: logP {ref_lp:.2f}, %N+O {mean(P_drug, a.ref, "pct_no"):.3f}, '
          f'HY/heavy {mean(P_drug, a.ref, "hy_h"):.4f}',
          '  가이던스를 끄면 조성이 native 쪽으로 이동하고 재현 지표가 좋아진다 = 조성 왜곡의 원인은 물리 손실이 아니라 가이던스.', '']

    # ---------------------------------------------------------------- 5
    L += [head(f'5. 교락 통제 — 크기 매칭 서브셋 (heavy {a.size[0]}-{a.size[1]})'), '']
    L += ['  소수성 접촉은 분자가 크고 기름지면 자동 증가한다. 크기를 맞춰도 결론이 유지되는지 확인.', '']
    L += [f'{"model":32s} {"n_pocket":>8s} {"HB/mol":>7s} {"HY/mol":>7s} {"IFP Tani":>9s} '
          f'{"HB F1":>7s} {"HY F1":>7s} {"logP":>6s} {"heavy":>6s}']
    L += [hr()]
    for t, lab, _ in order:
        if not P_size.get(t):
            continue
        L += [f'{lab:32s} {len(P_size[t]):8d} {fmt(mean(P_size, t, "hb"), 7, 2)} '
              f'{fmt(mean(P_size, t, "hy"), 7, 2)} {fmt(mean(P_size, t, "tani"), 9)} '
              f'{fmt(mean(P_size, t, "f1_hb"), 7)} {fmt(mean(P_size, t, "f1_hy"), 7)} '
              f'{fmt(mean(P_size, t, "logp"), 6, 2)} {fmt(mean(P_size, t, "heavy"), 6, 2)}']
    L += ['']

    # ---------------------------------------------------------------- 6
    L += [head('6. 특이성 음성대조 — 포켓 뒤섞기 (다른 포켓의 native IFP로 채점)'), '']
    dk = sorted(drug)
    L += [f'{"model":32s} {"IFP Tani":>9s} {"scrambled":>10s} {"ratio":>8s}']
    L += [hr()]
    for t, lab, _ in order:
        if not raw[t]:
            continue
        by = defaultdict(list)
        for r in raw[t]:
            by[r['pk']].append(r)
        sc = []
        for i, k in enumerate(dk):
            other = refs.get(dk[(i + 1) % len(dk)])
            if other is None or k not in by:
                continue
            rb = ifp(other)
            v = [tanimoto(ifp(r), rb) for r in by[k]]
            v = [x for x in v if not np.isnan(x)]
            if v:
                sc.append(np.mean(v))
        real = mean(P_drug, t, 'tani') if P_drug.get(t) else np.nan
        s = float(np.mean(sc)) if sc else np.nan
        rt = real / s if s and not np.isnan(s) and s > 0 else np.nan
        L += [f'{lab:32s} {fmt(real, 9)} {fmt(s, 10, 4)} {fmt(rt, 8, 1)}x']
    L += ['', '  전 모델이 동등하게 통과한다 = 건전성 검사이지 판별력 있는 지표가 아니다.', '']

    # ---------------------------------------------------------------- 7
    L += [head(f'7. 포켓 대응 유의성  (paired Wilcoxon, 기준 = {a.focus}, drug-like {len(drug)}포켓)'), '']
    L += ['  양수 = 기준 모델이 더 높음.  *** p<0.001, ** p<0.01, * p<0.05', '']
    METS = [('tani', 'IFP Tanimoto'), ('f1_hb', 'H-bond F1'), ('f1_hy', '소수성 F1'),
            ('rec_hb', 'H-bond recall'), ('rec_hy', '소수성 recall'),
            ('pre_hb', 'H-bond precision'), ('pre_hy', '소수성 precision'),
            ('hb', 'H-bond/mol'), ('hy', '소수성/mol')]
    L += [f'{"vs":32s} ' + ' '.join(f'{lab:>18s}' for _, lab in METS[:5])]
    L += [hr()]
    for t, lab, _ in order:
        if t == a.focus or not P_drug.get(t):
            continue
        cells = []
        for m, _ in METS[:5]:
            ks = sorted(set(P_drug[a.focus]) & set(P_drug[t]))
            n, d, p = paired([P_drug[a.focus][k][m] for k in ks], [P_drug[t][k][m] for k in ks])
            cells.append(f'{d:+14.4f}{sig(p)}' if not np.isnan(d) else f'{"-":>18s}')
        L += [f'{lab:32s} ' + ' '.join(cells)]
    L += ['']
    L += [f'{"vs":32s} ' + ' '.join(f'{lab:>18s}' for _, lab in METS[5:])]
    L += [hr()]
    for t, lab, _ in order:
        if t == a.focus or not P_drug.get(t):
            continue
        cells = []
        for m, _ in METS[5:]:
            ks = sorted(set(P_drug[a.focus]) & set(P_drug[t]))
            n, d, p = paired([P_drug[a.focus][k][m] for k in ks], [P_drug[t][k][m] for k in ks])
            cells.append(f'{d:+14.4f}{sig(p)}' if not np.isnan(d) else f'{"-":>18s}')
        L += [f'{lab:32s} ' + ' '.join(cells)]
    L += ['', hr('='),
          '해석 요령: recall이 유의하게 높은데 F1/Tanimoto가 n.s.이면 그 우위는 물량이지 배치가 아니다.',
          hr('=')]

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print(f'\n-> {a.out}')


if __name__ == '__main__':
    main()
