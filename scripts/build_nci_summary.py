#!/usr/bin/env python3
"""Non-covalent protein-ligand interactions per model, measured by BOTH instruments.

The question this answers: **how much non-covalent contact does each model's GENERATED pose make
with the protein**, on docking-successful molecules only.

Two independent tools are reported side by side because they do NOT measure the same thing and a
single number from either one is easy to over-read:

  PLIP 3.0.0        chemistry-typed, geometric, ANGLE-aware. H-bond needs d(D,A) < 4.1 A AND
                    angle(D-H...A) > 100 deg; hydrophobic needs a qualifying carbon pair < 4.0 A.
                    Counts are PLIP's own interaction objects (it dedups hydrophobic contacts per
                    ligand-atom x residue). Has pi-stacking / salt bridge / halogen / water bridge
                    as separate categories, and NO van-der-Waals category at all.
                    Source: eval_plip/<tag>/pocket*.jsonl  (scripts/plip_interactions.py)

  PoseCheck 1.3.1   wraps ProLIF `Fingerprint()`. A count is the number of (residue,
  (ProLIF)          interaction-type) CELLS that fire, so three contacts to one residue collapse to
                    one. Only four types ever fire here: HBAcceptor, HBDonor, Hydrophobic,
                    VdWContact -- and VdWContact, which PLIP does not have, is ~75% of the total.
                    Source: eval_out/<model>/posecheck/pocket*.csv  (scripts/eval_posecheck.py)

So PLIP's total and ProLIF's total are not comparable in LEVEL. They are comparable in RANKING,
and that is what section 3 checks: if the two disagree on the ordering of models, no interaction
claim should be made without saying which instrument produced it.

POPULATION: docking-successful molecules only, always. The reference .pt files (targetdiff, kgdiff,
pidiff) hold nothing else, so comparing against our full generated output would count different
things -- and the molecules the filter removes are exactly the fragmented ones. See
scripts/mark_docked.py and results/comparison/pose_quality_summary.txt.

Usage:
    python scripts/build_nci_summary.py --out results/comparison/f3_nci/nci_summary.txt
"""
import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

W = 150

# Roster comes from configs/models.json (Phase B, 2026-08-21). The tuple shape is unchanged --
# (label, PLIP tag under eval_plip/, model dir under eval_out/, source .pt, group) -- so the eight
# loops over MODELS below are untouched.
#
# Membership is families.nci.tables containing 'nci_summary'. F3 owns TWO tables over overlapping
# rosters (this one, 13 rows, and the IFP appendix, 19 rows, sharing only 7), which is why a list
# is needed and a single section string will not do.
#
# A None still means that instrument has no run for the model; the row prints '-' rather than
# being silently omitted, because a missing arm is itself something the reader must see.
def _registry_models(path='configs/models.json', extra=None):
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    import model_registry                                                   # noqa: E402
    rows = [m for m in model_registry.models(path)
            if model_registry.in_table(m, 'nci', 'nci_summary') or m['id'] in (extra or ())]
    rows.sort(key=lambda m: (m['families']['nci'].get('legacy_order', 10**6), m['id']))
    out = []
    for m in rows:
        ids = m.get('ids') or {}
        out.append((model_registry.label_for(m, 'nci'), ids.get('plip_tag'), ids.get('eval_out'),
                    model_registry.source_for(m, 'nci'), model_registry.group_for(m, 'nci')))
    return out


MODELS = _registry_models()

# Derived from configs/models.json: docked_policy == 'connected_is_docked'.
# Such a tag's .pt/SDF was rebuilt from a --samples run via load_samples, which keeps only
# single-fragment molecules. By the verified docked<=>connected equivalence that set IS the
# docking-successful set (2145 connected == 2145 with a Vina dock score in the consolidated .pt),
# so no vina-based filter is needed or possible -- every measured molecule is already
# docking-successful. Reported docked-only, NOT flagged [undocked]. See docked-iff-connected.
def _connected_is_docked(path='configs/models.json'):
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    import model_registry                                                   # noqa: E402
    return {(m.get('ids') or {}).get('plip_tag') for m in model_registry.models(path)
            if m.get('docked_policy') == 'connected_is_docked'} - {None}


CONNECTED_IS_DOCKED = _connected_is_docked()

PLIP_TYPES = [('n_hbond', 'H-bond'), ('n_hydrophobic', 'hydroph'), ('n_pistack', 'pi-stack'),
              ('n_saltbridge', 'salt br'), ('n_halogen', 'halogen'), ('n_waterbridge', 'water br')]
PROLIF_TYPES = [('int_HBAcceptor', 'HBAcc'), ('int_HBDonor', 'HBDon'),
                ('int_Hydrophobic', 'hydroph'), ('int_VdWContact', 'VdW')]


def docked_keys(pt_path):
    """Return (keep, heavy) for a sampling .pt. Delegates to scripts/docked.py.

    The body used to live here, in build_interaction_tables.py, in aggregate_posecheck.py and in
    build_gnina_comparison.py -- four copies that had drifted apart. docked.py holds the verbatim
    transplant of each; scripts/check_docked_equivalence.py proves this call returns exactly what
    the local copy returned, on every model .pt.

    Policy is 'dock_or_minimize' -- the one this table has always used, falling back to the
    minimize affinity for a .pt that never ran the dock mode. aggregate_posecheck uses 'dock'.
    The two are NOT interchangeable, which is why the policy is named rather than defaulted.
    """
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    import docked as _docked                                                # noqa: E402
    return _docked.docked_keys(pt_path, policy='dock_or_minimize')


def canonical_plip_key(canon_pt='results/sampling_results/targetdiff_vina_docked.pt'):
    """Row -> p<NNN>_m<MMMM> in CANONICAL pocket order, keyed off the full ligand filename.

    An eval_plip/ export numbers its pockets the way its source .pt was enumerated. For a GROUPED
    .pt that is already the canonical order, but a FLAT one is enumerated by order of appearance:
    PIDiff's export calls canonical pocket 47 "46" and is shifted from there on, 48 of its 99
    pockets. Every row still names its own receptor correctly -- nothing is mis-measured -- but the
    key string means a different molecule than the same string does in the .pt.
    """
    import sys as _s, os as _o
    _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
    import build_comparison_tables as _bct                                  # noqa: E402
    obj = _bct.load_pt(canon_pt)
    _, full2idx, dir2idx = _bct.build_pocket_maps(_bct.canonical_index(obj))
    del obj

    def key(r):
        i = full2idx.get(r.get('pk'))
        if i is None:
            i = dir2idx.get(str(r.get('pk', '')).split('/')[0])
        return None if i is None else f"p{i:03d}_m{int(r['mi']):04d}"
    return key


def load_plip(tag, keep, key=None):
    """keep=None means no docking filter available (undocked model) -> keep every measured row.

    `key` maps a row to the molecule key the `keep` test uses. It defaults to the export's OWN
    p<pocket_idx>_m<mi>, which is wrong whenever that export numbers pockets differently from the
    .pt `keep` came from -- `keep` is always in canonical space. On PIDiff the two spaces differ,
    and testing raw keys against canonical ones silently kept an arbitrary 804 of the 850
    molecules PLIP had in fact measured. main() passes canonical_plip_key() so both sides of the
    test live in one space; for every other model the two spaces coincide and the result is
    unchanged (audited across all 14 rows).
    """
    rows = []
    for p in sorted(glob.glob(os.path.join('eval_plip', tag, 'pocket*.jsonl'))):
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                if keep is None:
                    rows.append(r)
                    continue
                k = key(r) if key else f"p{int(r['pocket_idx']):03d}_m{int(r['mi']):04d}"
                if k is not None and k in keep:
                    rows.append(r)
    return rows


def load_prolif_contacts(model_dir):
    """{molecule: {'<type>_contact': int}} from prolif_counts/ (count=True). Empty if not run."""
    out = {}
    for p in sorted(glob.glob(os.path.join('eval_out', model_dir, 'prolif_counts', 'pocket*.csv'))):
        for r in csv.DictReader(open(p)):
            n = r.get('molecule')
            if n:
                out[n] = {k: float(v) for k, v in r.items()
                          if k.endswith('_contact') and v not in (None, '')}
    return out


def load_prolif(model_dir, keep, heavy):
    """Rows for the ProLIF arm.

    Normally driven by eval_out/<dir>/posecheck/pocket*.csv, which carries the count=False
    (per-residue boolean) `int_*` columns plus n_interactions, with the count=True contact totals
    joined in from prolif_counts/.

    A DOCKED-POSE tree (eval_out/<tag>_<engine>_dock/, built from the engine's own SDF) has no
    posecheck/ run -- PoseCheck on 26 model x engine combinations is ~3900 core-h, which is not
    affordable. It does not need one: eval_prolif_counts.py writes the same boolean convention as
    `<type>_res` alongside `<type>_contact`, and the two agree exactly (audited 2026-09-08 on
    targetdiff pocket000: int_HBAcceptor==HBAcceptor_res and the other three types, 84/84 each).
    So when posecheck/ is missing we drive from prolif_counts/ and map <type>_res -> int_<type>.

    n_interactions is recovered the same way: it is the count of fingerprint cells that fired,
    which is the sum of the four <type>_res values (verified 84/84 on targetdiff pocket000). So a
    docked-pose row carries no value sourced from the generated pose.
    """
    contacts = load_prolif_contacts(model_dir)
    pc_glob = sorted(glob.glob(os.path.join('eval_out', model_dir, 'posecheck', 'pocket*.csv')))
    if not pc_glob:
        rows = []
        for p in sorted(glob.glob(os.path.join('eval_out', model_dir,
                                               'prolif_counts', 'pocket*.csv'))):
            for r in csv.DictReader(open(p)):
                n = r.get('molecule')
                if not n or (keep is not None and n not in keep):
                    continue
                rec = {'mol': n, 'heavy': heavy.get(n), 'pocket_idx': int(n[1:4])}
                for c, _ in PROLIF_TYPES:
                    v = r.get(c.replace('int_', '') + '_res')
                    rec[c] = float(v) if v not in (None, '') else 0.0
                # n_interactions in the posecheck CSV is the number of fingerprint cells that
                # fired, i.e. exactly the sum of the four <type>_res columns. Verified on
                # targetdiff pocket000: 84/84 molecules agree to the last digit. So the docked
                # table needs nothing from the generated-pose PoseCheck run -- every column here
                # is measured on the docked pose.
                rec['n_interactions'] = sum(rec[c] for c, _ in PROLIF_TYPES)
                ct = contacts.get(n)
                rec['hydroph_contact'] = (ct.get('Hydrophobic_contact', 0.0)
                                          if ct is not None else None)
                rows.append(rec)
        return rows
    rows = []
    for p in pc_glob:
        for r in csv.DictReader(open(p)):
            n = r.get('molecule')
            if not n or (keep is not None and n not in keep) or not r.get('n_interactions'):
                continue
            # 'mol' is additive and unused here; scripts/build_nci_strict.py needs the molecule
            # key to intersect this arm with the PLIP arm, and reconstructing it there would mean
            # a second copy of this loader.
            rec = {'mol': n, 'heavy': heavy.get(n), 'pocket_idx': int(n[1:4])}  # pNNN_mMMMM -> pocket
            for c, _ in PROLIF_TYPES:
                rec[c] = float(r[c]) if r.get(c) not in (None, '') else 0.0
            rec['n_interactions'] = float(r['n_interactions'])
            # per-contact (count=True) hydrophobic, joined by molecule name. A molecule that WAS
            # measured but made zero hydrophobic contacts has an entry with no Hydrophobic_contact
            # key -> count it as 0.0, NOT None; else the mean is taken over only the molecules that
            # HAVE hydrophobic contacts and is biased upward (bit NATIVE hard: 1 mol/pocket, many 0s).
            # None only when the molecule is absent from prolif_counts entirely (genuinely unmeasured).
            ct = contacts.get(n)
            rec['hydroph_contact'] = (ct.get('Hydrophobic_contact', 0.0) if ct is not None else None)
            rows.append(rec)
    return rows


def col(rows, c):
    v = [r[c] for r in rows if r.get(c) is not None]
    return np.array(v, dtype=float) if v else np.array([])


def m(rows, c):
    a = col(rows, c)
    return a.mean() if len(a) else np.nan


def f(x, w=8, p=2):
    return f'{x:{w}.{p}f}' if np.isfinite(x) else f'{"-":>{w}}'


def pocket_agg(rows, valfn, B=3000, seed=0):
    """Pocket-as-unit aggregation for a per-molecule metric.

    Molecules in a pocket share the receptor/site, so they are NOT independent (measured ICC ~0.6-0.7
    for these metrics -> molecule-level SE understates by ~7x). The pocket is the inferential unit.
    Returns dict with:
      mol_mean : mean pooled over ALL molecules (the descriptive number in the tables above)
      pk_mom   : mean over pockets of each pocket's molecule-MEAN   (equal weight per pocket)
      pk_moM   : mean over pockets of each pocket's molecule-MEDIAN
      lo, hi   : 95% CI of pk_mom by CLUSTER bootstrap (resample POCKETS, not molecules)
      n_pk     : number of pockets
    valfn(row) -> float or None (None skips the molecule).
    """
    by = defaultdict(list)
    for r in rows:
        v = valfn(r)
        if v is not None and np.isfinite(v):
            by[r['pocket_idx']].append(v)
    ks = [k for k in by if by[k]]
    if not ks:
        return dict(mol_mean=np.nan, pk_mom=np.nan, pk_moM=np.nan, lo=np.nan, hi=np.nan, n_pk=0)
    arr = {k: np.asarray(by[k], float) for k in ks}
    allv = np.concatenate([arr[k] for k in ks])
    pk_means = np.array([arr[k].mean() for k in ks])
    pk_meds = np.array([np.median(arr[k]) for k in ks])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ks), size=(B, len(ks)))
    boot = np.array([pk_means[idx[b]].mean() for b in range(B)])
    return dict(mol_mean=float(allv.mean()), pk_mom=float(pk_means.mean()),
                pk_moM=float(pk_meds.mean()), lo=float(np.percentile(boot, 2.5)),
                hi=float(np.percentile(boot, 97.5)), n_pk=len(ks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/comparison/f3_nci/nci_summary.txt')
    ap.add_argument('--extra_models', default='',
                    help='comma-separated registry ids to include beyond the nci_summary roster')
    ap.add_argument('--plip_suffix', default='',
                    help="e.g. '_vinardo_dock' -> reads eval_plip/<plip_tag><suffix>")
    ap.add_argument('--model_suffix', default='',
                    help="e.g. '_vinardo_dock' -> reads eval_out/<eval_out><suffix>/prolif_counts")
    a = ap.parse_args()
    if a.extra_models:
        # MODELS is built at import time from the nci_summary roster; recompute it so an extra
        # id can join the DOCKED-pose tables without a `tables` key (which would enrol it in the
        # pinned f3_nci roster and fail scripts/check_rosters.py).
        global MODELS
        MODELS = _registry_models(extra={x.strip() for x in a.extra_models.split(',') if x.strip()})

    plip_key = canonical_plip_key()
    data = {}
    for label, ptag, pdir, pt, grp in MODELS:
        keep, heavy = docked_keys(pt) if os.path.isfile(pt) else (set(), {})
        # samples-derived tag: its rebuilt .pt has no vina field (keep would be None), but the
        # molecules ARE the connected==docked set, so keep every one of them and report docked-only.
        if ptag in CONNECTED_IS_DOCKED and heavy:
            keep = set(heavy)
        ptag_s = f'{ptag}{a.plip_suffix}' if ptag else ptag
        pdir_s = f'{pdir}{a.model_suffix}' if pdir else pdir
        plip = load_plip(ptag_s, keep, plip_key) if ptag_s and os.path.isdir(f'eval_plip/{ptag_s}') else []
        prolif = load_prolif(pdir_s, keep, heavy) if pdir_s else []
        data[label] = {'plip': plip, 'prolif': prolif, 'grp': grp, 'ptag': ptag, 'pdir': pdir,
                       'n_docked': (len(keep) if keep is not None else None)}

    L, A = [], None
    A = L.append

    def dl(label):
        """Row label, marked when the model is on the full set (no docking filter yet)."""
        return f'{label} [undocked]' if data[label]['n_docked'] is None else label
    A('=' * W)
    A('NON-COVALENT PROTEIN-LIGAND INTERACTIONS OF THE GENERATED POSE')
    A('두 계측기(PLIP / PoseCheck-ProLIF)로 각각 측정, 도킹 성공 분자만')
    A('=' * W)
    A('')
    A('포즈: 모델이 생성한 좌표 그대로. 재도킹 포즈가 아니다.')
    A('모집단: 도킹 성공 분자만. 참조 .pt(targetdiff/kgdiff/pidiff)는 도킹 성공 분자만 담고 있어')
    A('        필터 없이 비교하면 우리 모델만 파편화 분자를 포함한 superset이 된다.')
    A('')
    A('두 계측기는 세는 대상이 다르다 -- 총량을 직접 빼서 비교하지 말 것:')
    A('  PLIP   : 화학 타이핑 + 기하 기준, H결합에 ANGLE 조건 있음(>100 deg). pi-stack/염다리/')
    A('           할로겐/물다리를 따로 센다. van der Waals 항목이 아예 없다.')
    A('  ProLIF : (잔기, 상호작용종) 셀 개수. 같은 잔기에 3번 접촉해도 1로 센다. VdWContact가')
    A('           있고 그것이 총량의 ~75%를 차지한다.')
    A('따라서 절대량이 아니라 모델 간 순위가 두 계측기에서 일치하는지를 봐야 한다 (섹션 3).')
    A('')
    A('-' * W)
    A('계측기 정의 (설치된 소스에서 직접 확인: plip 3.0.0 config / prolif 2.2.0 기본 Fingerprint)')
    A('-' * W)
    A('                        거리 컷오프            각도            집계 단위')
    A('  PLIP H-bond           D-A <= 4.1 A          D-H..A >= 100 deg    공여-수용 원자쌍마다')
    A('  ProLIF HBAcc/HBDon    D-A <= 3.5 A          D-H-A  130-180 deg   (잔기,종류) 셀당 bool')
    A('  PLIP hydrophobic      탄소쌍 <= 4.0 A       (없음)               원자쌍마다(리간드원자x잔기 1회 중복제거)')
    A('  ProLIF Hydrophobic    <= 4.5 A              (없음)               (잔기,종류) 셀당 bool')
    A('  ProLIF VdWContact     vdW반지름합+0.0 tol   (없음)               (잔기,종류) 셀당 bool  -- PLIP에는 대응 항목 없음')
    A('')
    A('  H-bond: ProLIF은 리간드 역할로만 쪼갠 것(HBAcc=리간드가 수용, HBDon=리간드가 공여)이라')
    A('          PLIP의 단일 hbond와 KIND는 같다. 개수 차이는 (1)각도 100 vs 130 deg (2)거리 4.1 vs')
    A('          3.5 A (3)원자쌍 vs 잔기-bool 세 가지에서 온다.')
    A('  Hydroph: 개수 격차는 거의 전부 집계 단위 차이다. 소수성 잔기(Leu/Ile/Phe...)는 탄소가')
    A('          많아 리간드 탄소 여럿이 4 A 안에 들어오는데, PLIP은 접촉마다 세고 ProLIF은 그 잔기')
    A('          전체를 1로 캡한다. 그래서 소수성에서 격차가 가장 크게 벌어진다.')
    A('  주의: PLIP이 "hydrophobic"으로 잡는 근접 접촉의 상당수가 ProLIF에선 Hydrophobic이 아니라')
    A('        VdWContact로 분류된다. PLIP hydroph <-> ProLIF Hydroph 를 1:1로 대응시키지 말 것.')
    A('')
    # Verified 2026-07-30 against MolCRAFT Table 4 (analysis/MolCraft.pdf p.15): same reference
    # ligand set (size 22.8 vs our 22.75), HBD matches (0.87 vs 0.83), but their hydrophobic is
    # 5.06 vs our 1.44 -- entirely the prolif 2.1.0 Hydrophobic redefinition. Kept in the generated
    # file, not just METHODS, because this is where a reader is tempted to cross-cite.
    A('  !!! ProLIF 열은 외부 논문 표와 직접 비교 불가 (버전 의존). 우리 계측기 =')
    A('      PoseCheck 1.3.1 + prolif 2.2.0. PoseCheck 는 prolif 를 무핀 의존(Requires-Dist: prolif)')
    A('      하고, prolif 2.1.0 이 Hydrophobic 정의에서 "N/O/F 에 붙은 방향족 탄소"를 배제하도록')
    A('      바꿨다. 같은 결정 리간드에 SMARTS 만 바꿔 재현: 1.44(>=2.1) / 2.04(pre-2.1) /')
    A('      11.20(prolif 1.0). MolCRAFT Table 4 의 Reference hydrophobic 5.06 은 그 중간 정의값이다')
    A('      (분자크기 22.8 vs 우리 22.75 로 데이터는 동일, HBD 0.87 vs 0.83 으로 파이프라인도 정상).')
    A('      vdW 차이(우리 9.12 vs 논문 6.61)는 preset 을 전부 시도해도 재현되지 않아 미해결로 둔다.')
    A('      외부 논문과 겨룰 때는 PLIP 열을 쓸 것. 상세: analysis/METHODS_interaction_metrics.md 6b.')
    A('')
    A('  실측 분해 (vina_fixed, 같은 분자 9597개 페어링, mean/mol):')
    A('    H-bond      PLIP 원자쌍 4.28 -> PLIP 잔기로 접기 3.57 -> ProLIF HBAcc+HBDon 2.46')
    A('                (4.28->3.57 = granularity,  3.57->2.46 = 더 엄격한 거리/각도).  spearman +0.55')
    A('    Hydroph     PLIP 원자쌍 7.00 -> PLIP 잔기로 접기 4.98 -> ProLIF Hydroph 1.51')
    A('                (거의 전부 granularity; ProLIF은 잔기당 1로 캡).  spearman +0.52')
    A('    참고: ProLIF VdWContact 9.28 -- PLIP에 없는 별도 범주라 소수성 근접 접촉의 일부를 흡수.')
    A('')
    A('  [PharDiff H-bond 주의] PharDiff 는 PLIP H-bond 이 4.70 으로 전 모델 최고지만 ProLIF HBAcc+HBDon')
    A('  은 2.39 로 중간이다. 이는 계측 오류가 아니라 실제 기하 차이다 (측정으로 분해 확인):')
    A('    (1) granularity 는 원인 아님 -- PLIP 원자쌍/잔기 비율이 1.21 로 타 모델과 동일. 잔기 단위로')
    A('        접어도 PharDiff 가 3.89 로 최고라 H-bond 접촉 자체는 진짜 가장 많다.')
    A('    (2) 원인은 ProLIF 의 더 엄격한 거리/각도 게이트다. PharDiff 의 H-bond 이 평균 d(D,A)=3.39 A 로')
    A('        전 모델 중 가장 길어(vina_fixed 3.28 / TargetDiff 3.33 / NATIVE 3.23), d<=3.5 A 비율이')
    A('        59.7% 에 그친다(vina_fixed 74.9%). 그래서 PLIP(d<4.1) 통과분의 상당수가 3.5-4.1 A 구간에')
    A('        있어 ProLIF(d<=3.5) 에서 탈락한다: ProLIF 게이트(d<=3.5 & 각도>=130) 통과율 36.2% vs')
    A('        vina_fixed 47.2%. HBDon=0.73 으로 정상이라 도너-H protonation 문제(DeepICL 류)도 아니다.')
    A('    => PharDiff(pharmacophore 유도)는 "H-bond 특징이 근처"는 충족하나 확산모델보다 느슨한(길고')
    A('       약간 휜) 기하로 배치한다. 관대한 PLIP 은 보상, 엄격한 ProLIF 은 감점 -- 두 계측기를 나란히')
    A('       봐야 하는 대표 사례. PharDiff 의 "PLIP H-bond 최고" 를 결합 강도로 과대해석하지 말 것.')
    A('')
    A('  *** 외부 논문(PoseCheck / FlexSBDD 등)의 hydrophobic 수치와 직접 비교 금지 ***')
    A('  이 표의 ProLIF 값은 plf.Fingerprint() 기본 count=False, 즉 (잔기,종류) 셀당 bool 이다.')
    A('  PoseCheck 원 논문과 FlexSBDD(Table 7, Appendix B.4)는 접촉당 개수(count=True raw-sum)로')
    A('  세어 hydrophobic ~5-6 을 보고한다. 같은 분자/같은 도구로 실측: count=False 2.36 vs')
    A('  count=True 5.29 (vina_fixed pocket0, 14mol) -- 약 3배 차이가 순전히 이 규약에서 온다.')
    A('  FlexSBDD 논문은 hydrophobic의 거리/원자 기준을 아예 주지 않고 PoseCheck에 위임한다')
    A('  (Appendix A.1: clash만 vdW합+0.5A tol 명시, HB/hydrophobic은 정성 서술뿐). 그러므로 우리')
    A('  ProLIF hydrophobic(count=False)을 그들의 ~6 옆에 놓으면 3배 낮아 보이는데 이는 방법 차이다.')
    A('')

    # ---------------------------------------------------------------- coverage
    A('=' * W)
    A('0. 측정 커버리지')
    A('=' * W)
    A('')
    A(f'{"model":26s}{"docked":>9s}{"PLIP n":>9s}{"손실":>8s}{"ProLIF n":>10s}{"손실":>8s}   '
      f'{"PLIP tag":26s}{"PoseCheck dir":24s}')
    A('-' * W)
    for label, *_ in MODELS:
        d = data[label]
        nd, np_, nr = d['n_docked'], len(d['plip']), len(d['prolif'])
        nds = f'{nd:9d}' if nd is not None else f'{"미도킹":>9s}'
        lp = f'{100*(nd-np_)/nd:6.1f}%' if nd and np_ else f'{"-":>7s}'
        lr = f'{100*(nd-nr)/nd:6.1f}%' if nd and nr else f'{"-":>7s}'
        A(f'{label:26s}{nds}{np_:9d}{lp:>8s}{nr:10d}{lr:>8s}   '
          f'{(d["ptag"] or "-"):26s}{(d["pdir"] or "-"):24s}')
    A('')
    A('docked = 그 .pt의 도킹 성공 분자 수.')
    if any(data[lb]['n_docked'] is None for lb, *_ in MODELS):
        A('`미도킹` = 소스 .pt에 Vina 결과가 아예 없어 도킹 성공 필터를 걸 수 없는 모델. 그 행만')
        A('측정된 전체 분자로 집계했고 아래 표에 [undocked] 로 표시되며 섹션3 순위에서 제외된다')
        A('-- 도킹 후 그 행의 pt를 vina_docked.pt로 바꾸면 다른 행과 같은 기준이 된다.')
    A('손실 = docking-successful 분자 중 그 도구가 결과를 못 낸 비율(타임아웃·전처리 실패).')
    A('0 이면 그 계측기로 아직 측정하지 않은 모델이다 -- 값이 아니라 공백으로 읽어야 한다.')
    A('')
    A('*** 이 손실은 MISSING AT RANDOM 이 아니다. PoseCheck 타임아웃은 strain 완화가 제한 시간')
    A('안에 수렴하지 못한 분자, 즉 기하가 병적인 분자다. 따라서 손실률이 큰 모델일수록 아래 값이')
    # Derived, never hardcoded: this sentence used to name a fixed 1%-8% range and silently went
    # stale when PharDiff joined with a 56% ProLIF shortfall (a cancelled posecheck array, not a
    # timeout), so the prose contradicted the table right above it.
    _loss = []
    for lb, *_ in MODELS:
        d = data[lb]
        nd, nr = d['n_docked'], len(d['prolif'])
        if nd and nr:
            _loss.append((100.0 * (nd - nr) / nd, lb))
    if _loss:
        lo, hi = min(_loss), max(_loss)
        A(f'낙관적으로 치우친다. 손실률이 모델마다 {lo[0]:.0f}%({lo[1]})에서 {hi[0]:.0f}%({hi[1]})까지')
        A('벌어지므로, 접촉 수 차이가 그 격차보다 작으면 결론을 내리지 말 것.')
    else:
        A('낙관적으로 치우친다. 접촉 수 차이가 손실률 격차보다 작으면 결론을 내리지 말 것.')
    # A shortfall this large is not a timeout pattern -- it means the run never finished. Such a row
    # is a biased subsample of pockets, so it must not be rank-compared against complete rows.
    _bad = [(v, lb) for v, lb in _loss if v > 10.0]
    if _bad:
        A('')
        A('!!! ProLIF 손실 10% 초과 = 타임아웃으로 설명되지 않는 수준이며, 그 행은 측정이 끝나지')
        A('    않은 것이다(포켓 일부만 채점된 편향 부분표본). 섹션 2/2b/3 에서 완주한 행과 나란히')
        A('    순위를 매기면 안 된다. 해당 모델의 posecheck 를 끝낸 뒤 재생성할 것:')
        for v, lb in sorted(_bad, reverse=True):
            A(f'      - {lb}: ProLIF 손실 {v:.0f}%')
    A('')
    A('이름 주의: PLIP 태그 `pidiff_exact` 는 PIDiff 논문 샘플이 아니라 PIDiff의 손실을 그대로')
    A('구현한 OUR 모델(head1_dock_342k)이다. 논문 배포본은 `pidiff_baseline` / eval_out/pidiff.')
    A('')

    # ---------------------------------------------------------------- 1 PLIP
    A('=' * W)
    A('1. PLIP — 분자당 비공유 상호작용 개수 (도킹 성공 분자)')
    A('=' * W)
    A('')
    A(f'{"model":30s}{"n":>7s}' + ''.join(f'{lab:>10s}' for _, lab in PLIP_TYPES)
      + f'{"TOTAL":>9s}{"heavy":>8s}{"TOT/heavy":>11s}')
    A('-' * W)
    for label, *_ in MODELS:
        r = data[label]['plip']
        if not r:
            A(f'{dl(label):30s}{0:7d}' + ''.join(f'{"-":>10s}' for _ in PLIP_TYPES)
              + f'{"-":>9s}{"-":>8s}{"-":>11s}')
            continue
        tot = sum(col(r, c) for c, _ in PLIP_TYPES)
        hv = col(r, 'heavy')
        A(f'{dl(label):30s}{len(r):7d}' + ''.join(f(m(r, c), 10) for c, _ in PLIP_TYPES)
          + f(tot.mean(), 9) + f(hv.mean(), 8) + f((tot / hv).mean(), 11, 4))
    A('')
    A('TOTAL = 여섯 종류의 합. TOT/heavy 는 분자별로 나눈 뒤 평균 -- 상호작용 수는 리간드 크기에')
    A('비례해서 늘기 때문에, 크기를 통제하지 않은 TOTAL 하나만 인용하면 안 된다.')
    A('')
    A('-' * W)
    A('PLIP 타입별 계수 단위 -- 외부 PLIP 논문(NCIDiff 등)과 비교할 때 반드시 확인')
    A('-' * W)
    A('PLIP은 타입마다 보고 단위가 다르다. 우리는 PLIP이 내놓는 상호작용 객체 수를 그대로 센다')
    A('(n_<type> = len(리스트)). 그 "단위"가 타입마다 원자쌍이기도, 그룹/고리이기도 하다:')
    A('')
    A('  타입          PLIP 보고 단위                         우리 값(예: TargetDiff)')
    A('  H-bond        공여-수용 원자쌍마다                    4.09')
    A('  hydrophobic   접촉마다(리간드원자 x 잔기 1회 중복제거) 4.82')
    A('  salt bridge   전하중심(그룹) 쌍마다 1개              0.47')
    A('  pi-stack      고리쌍마다 1개                         0.25')
    A('')
    A('NCIDiff (ICML 2024 workshop, results/papers/NCIdiff.pdf) 도 PLIP를 쓰고 PLIP 기본')
    A('임계값(H-bond<=4.1, hydrophobic<=4.0 A; 논문 Table 7)을 그대로 쓴다. 그런데 Table 1의')
    A('"# NCI edge" 를 우리 값과 대면 타입마다 결과가 갈린다 (둘 다 CrossDocked 100포켓 TargetDiff):')
    A('')
    A('                     NCIDiff Table1   우리 PLIP   일치?')
    A('  H-bond                   4.17          4.09     O  (Δ0.08)')
    A('  hydrophobic              4.40          4.82     O  (Δ0.4)')
    A('  salt bridge              2.44          0.47     X  (5배)')
    A('  pi-pi                    5.61          0.25     X  (22배)')
    A('')
    A('원인 = 그룹/고리 단위 타입의 원자쌍 전개. NCIDiff은 salt bridge/pi-pi 를 자기 모델의 bipartite')
    A('엣지 표현대로 "모티프 원자 x 단백질 원자 모든 쌍"으로 센다 (논문 Limitations 명시). PLIP이')
    A('원자 단위로 보고하는 H-bond/hydrophobic 은 두 방법이 일치하고, PLIP이 그룹/고리로 묶어')
    A('보고하는 salt bridge/pi-pi 만 어긋난다. 결정적 증거: NCIDiff의 Reference(결정 리간드) pi-pi')
    A('가 9.32 -- 실제 리간드가 분자당 9번 pi-스택은 불가능하며(PLIP-native로 0.29), 원자쌍 전개')
    A('여야만 나오는 값이다.')
    A('')
    A('=> 외부 PLIP 논문과 숫자를 나란히 놓는 비교는 H-bond / hydrophobic 에 한정하라. salt bridge')
    A('   /pi-pi 는 계수 단위가 달라 직접 비교 불가 -- 우리 값이 적은 것은 PLIP 표준 그룹 카운트라서다.')
    A('')
    A('아래는 salt bridge / pi-pi 를 NCIDiff 규약(원자쌍 전개 = |리간드모티프원자| x |단백질모티프원자|')
    A('의 합)으로 재집계한 값이다. 이 전개가 NCIDiff Reference 를 재현함을 확인했다: pi-pi 9.32=9.32')
    A('정확 일치, salt bridge 4.26 vs 4.81 근접(PLIP 전하그룹 정의 차이). (obj)=PLIP 표준 객체 수,')
    A('(edge)=원자쌍 전개. edge 만 NCIDiff Table 1 과 같은 기준이다.')
    A('')
    NCIDIFF = {'TargetDiff': (2.44, 5.61), 'NATIVE (crystal ligand)': (4.81, 9.32)}  # (SB, PP)
    A(f'{"model":26s}{"n":>7s}{"SB(obj)":>9s}{"SB(edge)":>10s}{"PP(obj)":>9s}{"PP(edge)":>10s}'
      f'{"  | NCIDiff Table1: SB":>22s}{"PP":>7s}')
    A('-' * W)
    for label, *_ in MODELS:
        r = data[label]['plip']
        if not r:
            continue
        sbo, sbe = m(r, 'n_saltbridge'), m(r, 'n_saltbridge_edges')
        ppo, ppe = m(r, 'n_pistack'), m(r, 'n_pistack_edges')
        nd = NCIDIFF.get(label)
        nds = f'{nd[0]:>21.2f}{nd[1]:>7.2f}' if nd else f'{"—":>21s}{"—":>7s}'
        A(f'{label:26s}{len(r):7d}{f(sbo,9)}{f(sbe,10)}{f(ppo,9)}{f(ppe,10)}  {nds}')
    A('')
    A('NCIDiff Table1 열은 TargetDiff / Reference(=NATIVE) 두 행만 우리와 같은 모델이라 채웠다.')
    A('검증 기준은 REFERENCE 다: 결정 리간드는 모두에게 동일한 고정 집합이라 값이 일치해야 하고,')
    A('실제로 PP 9.32=9.32 정확 일치(SB 4.26 vs 4.81 근접) -- 원자쌍 전개 공식이 맞음을 확정한다.')
    A('TargetDiff 행은 생성 샘플이라 논문마다 sampling 이 달라 정확히는 안 맞는다(우리 PP(edge)')
    A('8.10 vs NCIDiff 5.61; 우리 PP(obj) 0.25 vs 그들 ~0.17 -- 즉 파이스택 객체 수 자체가 달라서지')
    A('계수 규약 차이가 아니다). 요점: 이제 SB/PP 도 H-bond/hydrophobic 처럼 같은 "규약"이며, 절대')
    A('비교의 앵커는 Reference 다. (edge 가 "-" 면 새 필드로 PLIP 재실행 안 한 모델이다.)')
    A('')

    # ---------------------------------------------------------------- 2 ProLIF
    A('=' * W)
    A('2. PoseCheck / ProLIF — 분자당 상호작용 셀 개수 (도킹 성공 분자)')
    A('=' * W)
    A('')
    A('  hydroph 은 두 가지로 나란히 준다: (잔기) = 기본 count=False, 잔기당 bool (섹션1 주석의 그 값);')
    A('  (접촉) = count=True, 접촉당 개수 -- PoseCheck 원 논문 / FlexSBDD Table 7 과 같은 규약이라')
    A('  그들 수치(hydrophobic ~5-6)와 직접 비교하려면 이 칼럼을 써야 한다. n(접촉)은 prolif_counts/')
    A('  가 있는 분자 수(별도 count=True 재실행); "-" 는 아직 재집계 안 된 모델.')
    A('')
    hdr2 = ''.join(f'{lab:>10s}' if k != 'int_Hydrophobic' else f'{"hy(잔기)":>10s}'
                   for k, lab in PROLIF_TYPES)
    A(f'{"model":30s}{"n":>7s}' + hdr2
      + f'{"TOTAL":>9s}{"heavy":>8s}{"TOT/heavy":>11s}{"VdW제외":>9s}{"hy(접촉)":>10s}{"n(접촉)":>9s}')
    A('-' * W)
    for label, *_ in MODELS:
        r = data[label]['prolif']
        if not r:
            A(f'{dl(label):30s}{0:7d}' + ''.join(f'{"-":>10s}' for _ in PROLIF_TYPES)
              + f'{"-":>9s}{"-":>8s}{"-":>11s}{"-":>9s}{"-":>10s}{"-":>9s}')
            continue
        tot = col(r, 'n_interactions')
        hv = col(r, 'heavy')
        novdw = tot - col(r, 'int_VdWContact')
        hyc = col(r, 'hydroph_contact')          # only mols with a prolif_counts entry
        A(f'{dl(label):30s}{len(r):7d}' + ''.join(f(m(r, c), 10) for c, _ in PROLIF_TYPES)
          + f(tot.mean(), 9) + f(hv.mean(), 8) + f((tot / hv).mean(), 11, 4) + f(novdw.mean(), 9)
          + (f(hyc.mean(), 10) if len(hyc) else f'{"-":>10s}') + f'{len(hyc):9d}')
    A('')
    A('VdW제외 = TOTAL - VdWContact. PLIP에는 VdW 항목이 없으므로 섹션 1과 굳이 견주려면')
    A('이 칼럼이 그나마 가깝다 (그래도 셀 단위 vs 접촉 단위라 정확히 대응하지는 않는다).')
    A('hy(잔기) vs hy(접촉): 같은 fingerprint를 잔기당 bool 로 셀지 접촉당 개수로 셀지의 차이일')
    A('뿐이며, _res(=잔기) 값이 기존 posecheck int_Hydrophobic 과 분자단위로 일치함을 확인했다.')
    A('')

    # -------------------------------------------------- 2b POCKET-as-unit aggregation
    A('=' * W)
    A('2b. 포켓 단위 집계  (한 포켓 안 분자들은 같은 site 를 공유 -> 독립 아님)')
    A('=' * W)
    A('')
    A('위 섹션들의 평균은 전체 분자를 pooled 한 값이고 n=분자 수다. 그러나 같은 포켓 분자들은')
    A('같은 단백질/site 를 공유해 상관이 크다(이 지표들 측정 ICC ~0.6-0.7). 그래서 분자단위 SE 는')
    A('실제보다 ~7x 작고, 추론의 단위는 포켓(n~100)이다. 아래는 comparison_tables 방식과 같이 포켓을')
    A('단위로 재집계한다:')
    A('  mol-pooled = 전체 분자 평균(위 표의 값)')
    A('  포켓평균의평균 = 포켓마다 분자 MEAN 낸 뒤 포켓들로 평균 (포켓 동등가중) [95% CI]')
    A('  포켓중앙값의평균 = 포켓마다 분자 MEDIAN 낸 뒤 포켓들로 평균')
    A('  CI 는 포켓을 리샘플하는 CLUSTER bootstrap (분자 리샘플이 아님) -- redock 표와 동일 방식.')
    A('')

    PLIP_METRICS = [
        # ---- PLIP (분자당 상호작용 객체 수; salt bridge/pi-pi 는 obj, edge 는 섹션1 참조) ----
        ('PLIP H-bond',        'plip', lambda r: r.get('n_hbond')),
        ('PLIP hydrophobic',   'plip', lambda r: r.get('n_hydrophobic')),
        ('PLIP pi-stack(obj)', 'plip', lambda r: r.get('n_pistack')),
        ('PLIP salt br(obj)',  'plip', lambda r: r.get('n_saltbridge')),
        ('PLIP pi-pi(edge)',   'plip', lambda r: r.get('n_pistack_edges')),
        ('PLIP salt br(edge)', 'plip', lambda r: r.get('n_saltbridge_edges')),
        ('PLIP TOT/heavy',     'plip', lambda r: (sum(r.get(c, 0) for c, _ in PLIP_TYPES) / r['heavy'])
                                                 if r.get('heavy') else None),
        # ---- ProLIF (잔기당 bool 셀 수; hy(접촉) 만 count=True 접촉 수) ----
        ('ProLIF HBAcc',       'prolif', lambda r: r.get('int_HBAcceptor')),
        ('ProLIF HBDon',       'prolif', lambda r: r.get('int_HBDonor')),
        ('ProLIF hy(잔기)',    'prolif', lambda r: r.get('int_Hydrophobic')),
        ('ProLIF VdWContact',  'prolif', lambda r: r.get('int_VdWContact')),
        ('ProLIF TOTAL',       'prolif', lambda r: r.get('n_interactions')),
        ('ProLIF hy(접촉)',    'prolif', lambda r: r.get('hydroph_contact')),
        ('ProLIF TOT/heavy',   'prolif', lambda r: (r['n_interactions'] / r['heavy'])
                                                   if r.get('heavy') else None),
    ]
    for mlabel, src, valfn in PLIP_METRICS:
        A('-' * W)
        A(f'{mlabel}   (도킹 성공 분자, 포켓 단위)')
        A('-' * W)
        A(f'{"model":26s}{"n_pk":>5s}{"mol-pooled":>12s}{"포켓평균의평균":>16s}{"[95% CI]":>20s}'
          f'{"포켓중앙값평균":>16s}')
        for label, *_ in MODELS:
            rows = data[label][src]
            if not rows:
                continue
            s = pocket_agg(rows, valfn)
            if s['n_pk'] == 0:
                continue
            ci = f'[{s["lo"]:.3f}, {s["hi"]:.3f}]'
            A(f'{label:26s}{s["n_pk"]:5d}{s["mol_mean"]:12.3f}{s["pk_mom"]:16.3f}{ci:>20s}'
              f'{s["pk_moM"]:16.3f}')
        A('')
    A('읽는 법: "포켓평균의평균" 이 포켓을 동등가중한 추정치이고 그 [95% CI] 가 정직한 불확실성이다.')
    A('mol-pooled 와 포켓평균의평균 이 다르면 포켓마다 분자 수가 불균형하다는 뜻(큰 포켓이 mol-pooled')
    A('를 끌어당김). 모델 간 비교는 이 포켓값들로 paired Wilcoxon(포켓으로 짝) 해야 하며, 분자단위')
    A('n 으로 유의성을 주장하면 안 된다.')
    A('')

    # ---------------------------------------------------------------- 3 ranking
    A('=' * W)
    A('3. 두 계측기의 순위 일치 여부 (양쪽 다 측정된 모델만)')
    A('=' * W)
    A('')
    # A model on the full set (no docking filter) is on a different population, so it must NOT enter
    # the ranking / Spearman -- that would compare docked-only models against an undocked one.
    excluded = [lb for lb, *_ in MODELS
                if data[lb]['plip'] and data[lb]['prolif'] and data[lb]['n_docked'] is None]
    both = [lb for lb, *_ in MODELS
            if data[lb]['plip'] and data[lb]['prolif'] and data[lb]['n_docked'] is not None]
    if excluded:
        A('제외(미도킹, 모집단이 달라 순위 비교 불가): ' + ', '.join(excluded))
        A('')
    if len(both) < 2:
        A('양쪽 모두 측정된 모델이 2개 미만이다.')
    else:
        pl = {lb: sum(col(data[lb]['plip'], c) for c, _ in PLIP_TYPES).mean() for lb in both}
        pr = {lb: col(data[lb]['prolif'], 'n_interactions').mean() for lb in both}
        plh = {lb: (sum(col(data[lb]['plip'], c) for c, _ in PLIP_TYPES)
                    / col(data[lb]['plip'], 'heavy')).mean() for lb in both}
        prh = {lb: (col(data[lb]['prolif'], 'n_interactions')
                    / col(data[lb]['prolif'], 'heavy')).mean() for lb in both}
        rank = lambda d: {lb: i + 1 for i, lb in enumerate(sorted(d, key=lambda k: -d[k]))}  # noqa
        rp, rq, rph, rqh = rank(pl), rank(pr), rank(plh), rank(prh)
        A(f'{"model":26s}{"PLIP TOT":>10s}{"순위":>6s}{"ProLIF TOT":>12s}{"순위":>6s}'
          f'{"  |":>3s}{"PLIP/heavy":>12s}{"순위":>6s}{"ProLIF/heavy":>14s}{"순위":>6s}{"불일치":>8s}')
        A('-' * W)
        for lb in sorted(both, key=lambda k: rp[k]):
            A(f'{lb:26s}{pl[lb]:10.2f}{rp[lb]:6d}{pr[lb]:12.2f}{rq[lb]:6d}{"  |":>3s}'
              f'{plh[lb]:12.4f}{rph[lb]:6d}{prh[lb]:14.4f}{rqh[lb]:6d}'
              f'{abs(rph[lb] - rqh[lb]):8d}')
        from scipy.stats import spearmanr
        A('')
        A(f'Spearman(두 계측기 총량)      rho = {spearmanr(list(pl.values()), [pr[k] for k in pl]).correlation:+.3f}'
          f'   (n={len(both)} 모델)')
        A(f'Spearman(heavy로 나눈 값)     rho = {spearmanr(list(plh.values()), [prh[k] for k in plh]).correlation:+.3f}')
        A('')
        A('rho 가 높으면 "어느 모델이 접촉을 많이 만드는가"라는 결론은 계측기에 의존하지 않는다.')
        A('낮으면 어느 도구로 잰 값인지 반드시 밝히고, 한쪽만으로 주장하지 말아야 한다.')
    A('')

    A('=' * W)
    A('Source: eval_plip/<tag>/pocket*.jsonl   +   eval_out/<model>/posecheck/pocket*.csv')
    A('Regenerate: python scripts/build_nci_summary.py')
    A('관련: results/comparison/appendix/ifp/interaction_tables.txt (native 대비 재현 정확도 = IFP recall/F1),')
    A('      results/comparison/appendix/posebusters/pose_quality_summary.txt (clash / strain / PB-Valid)')
    A('=' * W)

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w') as fh:
        fh.write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print(f'\n-> {a.out}')
    _write_csv(data, os.path.splitext(a.out)[0] + '.csv')


def _write_csv(data, out):
    """Machine-readable twin of the report, one row per model.

    Carries the registry `id` as the join key -- a LABEL IS NOT A JOIN KEY. A metric the model
    was never measured on is left EMPTY, never 0.

    Counting note, and it matters for any downstream join: PLIP counts contacts at the ATOM-PAIR
    level while ProLIF caps at one per (residue, type). The two columns are NOT the same quantity
    and must never be summed or differenced across instruments. And ProLIF numbers are
    version-locked to prolif 2.2.0 -- do not cross-cite them against another paper.
    """
    import csv as _csv, sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    import model_registry                                                # noqa: E402

    ptag2id = {(m.get('ids') or {}).get('plip_tag'): m['id']
               for m in model_registry.models() if (m.get('ids') or {}).get('plip_tag')}

    def mean(rows, fn):
        vals = [fn(r) for r in rows]
        vals = [v for v in vals if v is not None]
        return round(float(np.mean(vals)), 4) if vals else ''

    header = (['id', 'label', 'group', 'plip_tag', 'eval_out', 'n_plip', 'n_prolif', 'n_docked',
               'heavy_mean']
              + [f'plip_{c}' for c, _ in PLIP_TYPES] + ['plip_total', 'plip_per_heavy']
              + [f'prolif_{c.replace("int_", "")}' for c, _ in PROLIF_TYPES]
              + ['prolif_total', 'prolif_per_heavy'])

    _os.makedirs(_os.path.dirname(_os.path.abspath(out)), exist_ok=True)
    with open(out, 'w', newline='') as fh:
        w = _csv.writer(fh)
        w.writerow(header)
        for label, ptag, pdir, pt, grp in MODELS:
            d = data.get(label) or {}
            plip, prolif = d.get('plip') or [], d.get('prolif') or []
            heavy_src = prolif or plip
            row = [ptag2id.get(ptag, ptag), label, grp, ptag, pdir,
                   len(plip), len(prolif),
                   '' if d.get('n_docked') is None else d['n_docked'],
                   mean(heavy_src, lambda r: r.get('heavy'))]
            # `.get(c, 0)`: a molecule with none of that interaction type has no key at all.
            # Reading it as missing instead of zero once inflated NATIVE hydrophobic 1.44 -> 4.00
            # by averaging over only the molecules that HAD the contact.
            row += [mean(plip, lambda r, _c=c: r.get(_c, 0)) for c, _ in PLIP_TYPES]
            row += [mean(plip, lambda r: sum(r.get(c, 0) for c, _ in PLIP_TYPES)),
                    mean(plip, lambda r: (sum(r.get(c, 0) for c, _ in PLIP_TYPES) / r['heavy'])
                         if r.get('heavy') else None)]
            row += [mean(prolif, lambda r, _c=c: r.get(_c, 0)) for c, _ in PROLIF_TYPES]
            row += [mean(prolif, lambda r: r.get('n_interactions')),
                    mean(prolif, lambda r: (r['n_interactions'] / r['heavy'])
                         if r.get('heavy') and r.get('n_interactions') is not None else None)]
            w.writerow(row)
    print(f'-> {out}')


if __name__ == '__main__':
    main()
