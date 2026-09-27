#!/usr/bin/env python3
"""Detailed PLIP / ProLIF NCI summary for the PHYSICS-TERM ABLATION.

Same instruments, definitions, conventions and pocket-unit statistics as
scripts/build_nci_summary.py (imported here, not re-implemented) — but over the ablation arms:
the 2^3 lattice of Vina term groups (steric / h-bond / hydrophobic) x guidance {ON, OFF}, anchored
by NATIVE (crystal ligand), no-physics (arm 1 = novdw) and all-three (arm 7 = vina_fixed).

Everything is on the docked (== connected) subset, on the GENERATED pose (not re-docked). PLIP/
ProLIF are functions of the generated coordinates, so they are unaffected by the exh-8 vs exh-16
docking difference between the ablation arms and the arm-1/7 reference rows; only the docked-only
FILTER matters, and docking failure == fragmentation regardless of exhaustiveness.

Ablation PLIP was measured via --samples (molecule index = result-file index), so its p###_m####
names do NOT match the merged-pt grouping; the ablation rows therefore skip the pt-name filter
(keep=None) — sound because --samples keeps only single-fragment == connected == docked molecules.
Reference rows (NATIVE / novdw / vina_fixed) keep the standard docked_keys pt filter.

Usage:
    python scripts/build_nci_summary_abl.py --out results/diagnostics/ablation_nci/nci_summary_abl.txt
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_nci_summary as B   # noqa: E402  (reuse loaders, aggregation, formatters, constants)

W = B.W

# label, PLIP tag (eval_plip/<tag>), ProLIF dir (eval_out/<dir>), source .pt, use_pt_filter
# use_pt_filter=False -> ablation row measured on the connected==docked set already (keep=None).
ABL = 'results/{t}/{t}_vina_docked.pt'.format
MODELS = [
    ('NATIVE (crystal ligand)',     'NATIVE', 'native',
     'results/sampling_results/crossdocked_test_vina_docked.pt', True),
    # reference endpoints of the lattice (guidance ON; generated pose comparable):
    # ===================== GUIDE ON (lattice order: 0 -> 1 -> 2 -> 3 terms) =====================
    ('arm1 no-physics (novdw) ON',  'novdw',  'novdw',
     'results/novdw_gen/novdw_gen_vina_docked.pt', True),
    ('arm2 vdW (S) ON',             'abl_vdw_on',        'abl_vdw_on',        ABL(t='abl_vdw_on'), False),
    ('arm3 h-bond (H) ON',          'abl_hbond_on',      'abl_hbond_on',      ABL(t='abl_hbond_on'), False),
    ('arm4 hydrophobic (P) ON',     'abl_hydro_on',      'abl_hydro_on',      ABL(t='abl_hydro_on'), False),
    ('arm5 vdW+h-bond (S+H) ON',    'abl_vdw_hbond_on',  'abl_vdw_hbond_on',  ABL(t='abl_vdw_hbond_on'), False),
    ('arm6 vdW+hydro (S+P) ON',     'abl_vdw_hydro_on',  'abl_vdw_hydro_on',  ABL(t='abl_vdw_hydro_on'), False),
    ('arm8 h-bond+hydro (H+P) ON',  'abl_hbond_hydro_on', 'abl_hbond_hydro_on', ABL(t='abl_hbond_hydro_on'), False),
    ('arm7 S+H+P (vina_fixed) ON',  'vina_fixed_best', 'vina_fixed',
     'results/vina_fixed_best_gen/vina_fixed_best_vina_docked.pt', True),
    # ===================== GUIDE OFF (lattice order) ============================================
    # arm1 OFF closes the lattice: generated 2026-09-15 from novdw's OWN checkpoint resampled with
    # no_guide, exactly how arm7 OFF was made from vina_fixed's. Provenance, coverage and the
    # per-arm comparison live in results/diagnostics/abl_none_off/RUN_RECORD.md.
    ('arm1 no-physics (novdw) OFF', 'abl_none_off',      'abl_none_off',      ABL(t='abl_none_off'), False),
    ('arm2 vdW (S) OFF',            'abl_vdw_off',       'abl_vdw_off',       ABL(t='abl_vdw_off'), False),
    ('arm3 h-bond (H) OFF',         'abl_hbond_off',     'abl_hbond_off',     ABL(t='abl_hbond_off'), False),
    ('arm4 hydrophobic (P) OFF',    'abl_hydro_off',     'abl_hydro_off',     ABL(t='abl_hydro_off'), False),
    ('arm5 vdW+h-bond (S+H) OFF',   'abl_vdw_hbond_off', 'abl_vdw_hbond_off', ABL(t='abl_vdw_hbond_off'), False),
    ('arm6 vdW+hydro (S+P) OFF',    'abl_vdw_hydro_off', 'abl_vdw_hydro_off', ABL(t='abl_vdw_hydro_off'), False),
    ('arm8 h-bond+hydro (H+P) OFF', 'abl_hbond_hydro_off', 'abl_hbond_hydro_off', ABL(t='abl_hbond_hydro_off'), False),
    # arm7 OFF = SAME vina_fixed (S+H+P) checkpoint sampled with no_guide (id0 log: guide_mode=
    # 'no_guide'). Measured via --samples -> connected==docked -> use_filter=False, like the abl OFF arms.
    ('arm7 S+H+P (vina_fixed) OFF', 'vina_fixed_best_noguide', 'vina_fixed_best_noguide',
     'results/noguide/vina_fixed_best_noguide/noguide_grouped.pt', False),
]

PLIP_TYPES, PROLIF_TYPES = B.PLIP_TYPES, B.PROLIF_TYPES
col, m, f, pocket_agg = B.col, B.m, B.f, B.pocket_agg

import csv, glob   # noqa: E402


def guide_of(label):
    """'ON' / 'OFF' / 'REF' (NATIVE) from the row label, for grouping headers."""
    if label.endswith(' ON'):
        return 'ON'
    if label.endswith('OFF'):
        return 'OFF'
    return 'REF'


def group_hdr(prev, cur):
    """A '--- guide ON/OFF ---' banner line when the guidance group changes, else None."""
    if cur != prev and cur in ('ON', 'OFF'):
        return f'-- guide {cur} ' + '-' * (W - len(f'-- guide {cur} '))
    return None


def load_prolif_full(model_dir, keep, heavy):
    """Like B.load_prolif but also carries PoseCheck clashes + strain_energy per molecule."""
    rows = B.load_prolif(model_dir, keep, heavy)
    extra = {}
    for p in sorted(glob.glob(os.path.join('eval_out', model_dir, 'posecheck', 'pocket*.csv'))):
        for r in csv.DictReader(open(p)):
            n = r.get('molecule')
            if not n:
                continue
            def num(k):
                v = r.get(k)
                try:
                    return float(v) if v not in (None, '') else None
                except ValueError:
                    return None
            extra[n] = (num('clashes'), num('strain_energy'))
    # rows from B.load_prolif are in posecheck file order; re-key by molecule via a second pass
    idx = 0
    for p in sorted(glob.glob(os.path.join('eval_out', model_dir, 'posecheck', 'pocket*.csv'))):
        for r in csv.DictReader(open(p)):
            n = r.get('molecule')
            if not n or (keep is not None and n not in keep) or not r.get('n_interactions'):
                continue
            if idx < len(rows):
                rows[idx]['clashes'], rows[idx]['strain_energy'] = extra.get(n, (None, None))
                idx += 1
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/diagnostics/ablation_nci/nci_summary_abl.txt')
    a = ap.parse_args()

    data = {}
    for label, ptag, pdir, pt, use_filter in MODELS:
        keep, heavy = B.docked_keys(pt) if os.path.isfile(pt) else (set(), {})
        plip_keep = keep if use_filter else None      # ablation PLIP came from --samples -> keep all
        plip = B.load_plip(ptag, plip_keep) if os.path.isdir(f'eval_plip/{ptag}') else []
        prolif_keep = keep if use_filter else None
        prolif = load_prolif_full(pdir, prolif_keep, heavy) if os.path.isdir(f'eval_out/{pdir}') else []
        n_dock = len(keep) if (use_filter and keep is not None) else (len(plip) or None)
        data[label] = {'plip': plip, 'prolif': prolif, 'ptag': ptag, 'pdir': pdir, 'n_docked': n_dock}

    L = []
    A = L.append
    A('=' * W)
    A('PHYSICS-TERM ABLATION — 비공유 상호작용 (PLIP / PoseCheck-ProLIF), 도킹 성공 분자만')
    A('=' * W)
    A('')
    A('격자: Vina 항 그룹 { steric S = gauss1+gauss2+repulsion,  h-bond H,  hydrophobic P }의')
    A('      2^3 부분집합 x guidance {ON=head1_only 100/25, OFF=no_guide}. 24 mol/pocket x 100 pocket.')
    A('앵커: NATIVE(결정 리간드), arm1 no-physics(novdw), arm7 S+H+P(vina_fixed).')
    A('')
    A('포즈: 생성 좌표 그대로(재도킹 아님) -> PLIP/ProLIF 값은 도킹 exhaustiveness와 무관.')
    A('모집단: 도킹 성공(==연결된) 분자만. 물리항 인력만으로 파편화된 분자는 도킹 실패로 빠지며,')
    A('        그 파편화율 자체가 결과다(아래 conn% 및 RESULTS_physics_term_ablation 참조).')
    A('계측기 정의/단위/외부논문 비교 주의는 results/comparison/f3_nci/nci_summary.txt 와 동일(그 파일 참조).')
    A('두 계측기는 총량 LEVEL이 아니라 RANKING으로 비교(섹션 3).')
    A('')

    # -------- 0. coverage --------
    A('=' * W)
    A('0. 측정 커버리지')
    A('=' * W)
    A('')
    A(f'{"model":30s}{"docked":>8s}{"PLIP n":>9s}{"ProLIF n":>10s}{"손실":>8s}   {"tag":26s}')
    A('-' * W)
    for label, ptag, pdir, pt, _ in MODELS:
        d = data[label]
        nd, np_, nr = d['n_docked'], len(d['plip']), len(d['prolif'])
        nds = f'{nd:8d}' if nd else f'{"-":>8s}'
        lr = f'{100*(np_-nr)/np_:6.1f}%' if np_ and nr else f'{"-":>7s}'
        A(f'{label:30s}{nds}{np_:9d}{nr:10d}{lr:>8s}   {ptag:26s}')
    A('')
    A('docked = 도킹 성공(==연결) 분자 수. PLIP n = 그 분자 중 PLIP이 잰 수(-samples는 연결분자만).')
    A('ProLIF n = PoseCheck가 잰 수; 손실 = PLIP 대비 ProLIF가 못 낸 비율(strain 타임아웃 등, MNAR).')
    A('')

    # -------- 1. PLIP per-molecule --------
    A('=' * W)
    A('1. PLIP — 분자당 비공유 상호작용 개수 (도킹 성공 분자)')
    A('=' * W)
    A('')
    A(f'{"model":30s}{"n":>7s}' + ''.join(f'{lab:>10s}' for _, lab in PLIP_TYPES)
      + f'{"TOTAL":>9s}{"heavy":>8s}{"TOT/heavy":>11s}')
    A('-' * W)
    prev = 'REF'
    for label, *_ in MODELS:
        gh = group_hdr(prev, guide_of(label)); prev = guide_of(label)
        if gh:
            A(gh)
        r = data[label]['plip']
        if not r:
            A(f'{label:30s}{0:7d}' + ''.join(f'{"-":>10s}' for _ in PLIP_TYPES)
              + f'{"-":>9s}{"-":>8s}{"-":>11s}')
            continue
        tot = sum(col(r, c) for c, _ in PLIP_TYPES)
        hv = col(r, 'heavy')
        A(f'{label:30s}{len(r):7d}' + ''.join(f(m(r, c), 10) for c, _ in PLIP_TYPES)
          + f(tot.mean(), 9) + f(hv.mean(), 8) + f((tot / hv).mean(), 11, 4))
    A('')
    A('TOTAL = 여섯 종류 합. TOT/heavy = 분자별로 나눈 뒤 평균(크기 통제). H-bond/hydrophobic 은')
    A('원자쌍 단위, salt bridge/pi-stack 은 그룹/고리 객체 단위 -- 외부 PLIP 논문과는 H-bond/')
    A('hydrophobic 만 직접 비교 가능(자세히는 nci_summary.txt 섹션 1).')
    A('')

    # -------- 2. ProLIF per-molecule --------
    A('=' * W)
    A('2. PoseCheck / ProLIF — 분자당 상호작용 셀 개수 (도킹 성공 분자)')
    A('=' * W)
    A('')
    A('  hy(잔기)=count=False 잔기당 bool;  hy(접촉)=count=True 접촉당 개수(PoseCheck/FlexSBDD 규약).')
    A('  VdWContact는 PLIP에 없는 범주이며 ProLIF TOTAL의 ~75%.')
    A('')
    hdr2 = ''.join(f'{lab:>10s}' if k != 'int_Hydrophobic' else f'{"hy(잔기)":>10s}'
                   for k, lab in PROLIF_TYPES)
    A(f'{"model":30s}{"n":>7s}' + hdr2
      + f'{"TOTAL":>9s}{"heavy":>8s}{"TOT/heavy":>11s}{"VdW제외":>9s}{"hy(접촉)":>10s}{"clash":>8s}{"strain(med)":>12s}')
    A('-' * W)
    prev = 'REF'
    for label, *_ in MODELS:
        gh = group_hdr(prev, guide_of(label)); prev = guide_of(label)
        if gh:
            A(gh)
        r = data[label]['prolif']
        if not r:
            A(f'{label:30s}{0:7d}' + ''.join(f'{"-":>10s}' for _ in PROLIF_TYPES)
              + f'{"-":>9s}{"-":>8s}{"-":>11s}{"-":>9s}{"-":>10s}{"-":>8s}{"-":>12s}')
            continue
        tot = col(r, 'n_interactions'); hv = col(r, 'heavy')
        novdw = tot - col(r, 'int_VdWContact'); hyc = col(r, 'hydroph_contact')
        cl = col(r, 'clashes'); st = col(r, 'strain_energy')
        A(f'{label:30s}{len(r):7d}' + ''.join(f(m(r, c), 10) for c, _ in PROLIF_TYPES)
          + f(tot.mean(), 9) + f(hv.mean(), 8) + f((tot / hv).mean(), 11, 4) + f(novdw.mean(), 9)
          + (f(hyc.mean(), 10) if len(hyc) else f'{"-":>10s}')
          + (f(cl.mean(), 8) if len(cl) else f'{"-":>8s}')
          + (f(np.median(st), 12) if len(st) else f'{"-":>12s}'))
    A('')
    A('clash=mean, strain=MEDIAN(kcal/mol). *** strain 원값은 극단적 heavy-tail(개별 최대 ~1e15,')
    A('UFF 완화 발산) 이라 mean 은 1e13 규모로 무의미 -- median 만 인용한다. 이 median 값은 캐논')
    A('scripts/aggregate_posecheck.py / posecheck_summary.txt 와 분자단위로 동일하다(arm7 ON 565.67 일치).')
    A('생성 분자의 내부 strain 은 이 확산모델群에서 원래 크다: 참조로 TargetDiff 179 / KGDiff 644 /')
    A('vina_fixed 565 / novdw 440, 결정 리간드 NATIVE 는 9 (이미 이완된 상태). 즉 절대값 크기가 아니라')
    A('arm 간 상대 비교로 읽어야 한다. strain 타임아웃(t/o)은 기하가 병적인 분자에 몰려(MNAR) 손실률이')
    A('큰 arm 일수록 낙관 편향 -- conn% 낮은 arm 특히 주의.')
    A('')

    # -------- 2b. pocket-unit --------
    A('=' * W)
    A('2b. 포켓 단위 집계 (같은 포켓 분자는 site 공유 -> 독립 아님; 추론 단위는 포켓)')
    A('=' * W)
    A('  포켓평균의평균 [95% CI(cluster bootstrap, 포켓 리샘플)] + 포켓중앙값평균. mol-pooled=위 표 값.')
    A('')
    METRICS = [
        ('PLIP H-bond',       'plip',   lambda r: r.get('n_hbond')),
        ('PLIP hydrophobic',  'plip',   lambda r: r.get('n_hydrophobic')),
        ('PLIP TOT/heavy',    'plip',   lambda r: (sum(r.get(c, 0) for c, _ in PLIP_TYPES) / r['heavy'])
                                                  if r.get('heavy') else None),
        ('ProLIF HBAcc+HBDon', 'prolif', lambda r: (r.get('int_HBAcceptor', 0) + r.get('int_HBDonor', 0))),
        ('ProLIF hy(잔기)',   'prolif', lambda r: r.get('int_Hydrophobic')),
        ('ProLIF hy(접촉)',   'prolif', lambda r: r.get('hydroph_contact')),
        ('ProLIF VdWContact', 'prolif', lambda r: r.get('int_VdWContact')),
        ('ProLIF TOT/heavy',  'prolif', lambda r: (r['n_interactions'] / r['heavy'])
                                                  if r.get('heavy') else None),
        ('PoseCheck clashes', 'prolif', lambda r: r.get('clashes')),
        ('PoseCheck strain',  'prolif', lambda r: r.get('strain_energy')),
    ]
    for mlabel, src, valfn in METRICS:
        A('-' * W)
        A(f'{mlabel}   (도킹 성공 분자, 포켓 단위)')
        A('-' * W)
        A(f'{"model":30s}{"n_pk":>5s}{"mol-pooled":>12s}{"포켓평균의평균":>16s}{"[95% CI]":>22s}'
          f'{"포켓중앙값평균":>16s}')
        for label, *_ in MODELS:
            rows = data[label][src]
            if not rows:
                continue
            s = pocket_agg(rows, valfn)
            if s['n_pk'] == 0:
                continue
            ci = f'[{s["lo"]:.3f}, {s["hi"]:.3f}]'
            A(f'{label:30s}{s["n_pk"]:5d}{s["mol_mean"]:12.3f}{s["pk_mom"]:16.3f}{ci:>22s}'
              f'{s["pk_moM"]:16.3f}')
        A('')

    # -------- 3. ranking agreement --------
    A('=' * W)
    A('3. 두 계측기 순위 일치 (양쪽 다 측정된 모델)')
    A('=' * W)
    A('')
    both = [lb for lb, *_ in MODELS if data[lb]['plip'] and data[lb]['prolif']]
    if len(both) >= 2:
        pl = {lb: sum(col(data[lb]['plip'], c) for c, _ in PLIP_TYPES).mean() for lb in both}
        pr = {lb: col(data[lb]['prolif'], 'n_interactions').mean() for lb in both}
        plh = {lb: (sum(col(data[lb]['plip'], c) for c, _ in PLIP_TYPES)
                    / col(data[lb]['plip'], 'heavy')).mean() for lb in both}
        prh = {lb: (col(data[lb]['prolif'], 'n_interactions')
                    / col(data[lb]['prolif'], 'heavy')).mean() for lb in both}
        rank = lambda d: {lb: i + 1 for i, lb in enumerate(sorted(d, key=lambda k: -d[k]))}  # noqa
        rp, rq, rph, rqh = rank(pl), rank(pr), rank(plh), rank(prh)
        A(f'{"model":30s}{"PLIP TOT":>10s}{"순위":>6s}{"ProLIF TOT":>12s}{"순위":>6s}'
          f'{"  |":>3s}{"PLIP/heavy":>12s}{"순위":>6s}{"ProLIF/heavy":>14s}{"순위":>6s}{"불일치":>8s}')
        A('-' * W)
        for lb in sorted(both, key=lambda k: rp[k]):
            A(f'{lb:30s}{pl[lb]:10.2f}{rp[lb]:6d}{pr[lb]:12.2f}{rq[lb]:6d}{"  |":>3s}'
              f'{plh[lb]:12.4f}{rph[lb]:6d}{prh[lb]:14.4f}{rqh[lb]:6d}{abs(rph[lb]-rqh[lb]):8d}')
        from scipy.stats import spearmanr
        A('')
        A(f'Spearman(총량)     rho = {spearmanr(list(pl.values()), [pr[k] for k in pl]).correlation:+.3f}   (n={len(both)})')
        A(f'Spearman(heavy)   rho = {spearmanr(list(plh.values()), [prh[k] for k in plh]).correlation:+.3f}')
    A('')
    A('=' * W)
    A('Source: eval_plip/<tag>/pocket*.jsonl + eval_out/<tag>/{posecheck,prolif_counts}/pocket*.csv')
    A('Regenerate: python scripts/build_nci_summary_abl.py')
    A('관련: analysis/RESULTS_physics_term_ablation_2026-07.md, results/comparison/f3_nci/nci_summary.txt')
    A('=' * W)

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w') as fh:
        fh.write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print(f'\n-> {a.out}')


if __name__ == '__main__':
    main()
