# NCI Benchmark — 데이터 출처 (provenance)

**작성**: 2026-07-24 | **canonical 포켓 목록**: `analysis/nci_benchmark/test_pockets.json`
sha256(ordered ligand_filename list) = `fa4a49fe58ebbaf588d031f73b907857d6ad0a68e4d6736e16ce5d4d47d48421`

이 파일은 표의 각 행이 **어디서 왔는지**를 기록한다. 논문 캡션은 이 문서에 근거해야 하며,
여기 적히지 않은 출처 주장은 표에 쓰지 않는다.

---

## 1. 핵심 사실 — 베이스라인은 재생성한 것이 아니라 재평가한 것이다

원 계획서(§7.2)는 캡션에 *"All baselines were **regenerated** in-house"*라고 쓰라고 했다.
**이 데이터에서는 거짓이다.** targetdiff / kgdiff / PIDiff / IPDiff는 전부 공개 릴리스 샘플이며,
우리가 그 모델을 다시 돌린 것이 아니다. `METHODS_interaction_metrics.md` §7.5도 이미
"코드·시드·체크포인트 미기록의 외부 산물"이라고 적어 두었다.

우리가 실제로 통일한 것은 **생성**이 아니라 **평가**다. 그 자체로 충분히 의미 있는 기여다:
동일 100포켓, 동일 protonation 백엔드, 동일 계측기, 동일 분모(도킹 성공 분자), 동일 통계 절차.

### 캡션에 쓸 문장 (승인된 문안)

> All models are compared on the same 100 CrossDocked2020 test pockets (Luo et al., 2021 split;
> pocket ordering verified by checksum against every source file) and were **evaluated with an
> identical in-house protocol**: one shared interaction engine, one shared protonation backend per
> instrument, and a single shared denominator of docking-successful molecules. Generated samples
> for TargetDiff, KGDiff, PIDiff and IPDiff are the authors' public releases rather than in-house
> re-generations; no metric value is copied from any publication. Interactions are computed with
> PoseCheck (ProLIF) on the raw generated poses without redocking; PLIP-based results appear in
> Appendix Table S_x.

**"논문에서 수치를 베끼지 않는다"는 원칙(§0-2)은 그대로 유효하다** — 모든 셀은 우리 파이프라인
산출값이다. 바뀐 것은 "샘플을 우리가 만들었다"는, 검증 불가능한 주장뿐이다.

---

## 2. IPDiff ≠ PIDiff

이름이 비슷하지만 **다른 모델**이다. 표에서 절대 합치지 말 것.

| | IPDiff | PIDiff |
|---|---|---|
| 정식 명칭 | Interaction Prior Diffusion | Physics-Informed Diffusion |
| 저자 | YangLing0818 | 별개 그룹 |
| 이 프로젝트에서의 역할 | SOTA 비교군 | 우리가 재현한 vdW 손실의 원논문 |
| in-repo 재현본 | 없음 | 있음 (`pidiff_exact`) |

---

## 3. 모델별 소스 (실측 확인, 2026-07-24)

"도킹" 컬럼은 `mol`이 3D 좌표를 가진 분자 중 `vina.dock[0].affinity is not None`인 비율이다.

| 표의 행 | eval_out 태그 | eval_plip 태그 | 소스 `.pt` | 출처 | 3D 분자 | 도킹 성공 |
|---|---|---|---|---|---|---|
| Reference | *(생성 예정)* | `NATIVE` | `results/sampling_results/crossdocked_test_vina_docked.pt` | CrossDocked 결정 리간드 | 100 | 100 (100%) |
| TargetDiff | `targetdiff` | `targetdiff` | `results/sampling_results/targetdiff_vina_docked.pt` | 공개 릴리스 | 9036 | 9036 (100%) |
| KGDiff | `kgdiff` | `kgdiff` | `results/sampling_results/our_vina_score_docked.pt` | 공개 릴리스 | 8813 | 8813 (100%) |
| PIDiff (공식) | *(생성 예정)* | *(생성 예정)* | `results/sampling_results/PIDiff_vina_docked_complete.pt` | 공개 릴리스 | 850 | 850 (100%) |
| PIDiff (in-repo 재현) | `pidiff` | `pidiff_exact` | `results/head1_dock_342k/head1_dock_342k_vina_docked.pt` | in-repo | 8213 | 8213 (100%) |
| IPDiff | *(생성 예정)* | *(생성 예정)* | `results/sampled_results_ipdiff_official/result_{0..99}.pt` | 공개 릴리스 | 10000 | **도킹 정보 없음** |
| IRDiff | — | — | — | **미생성 (연기)** | — | — |
| **Ours** (vina loss) | `vina_fixed` | `vina_fixed_best` | `results/vina_fixed_best_gen/vina_fixed_best_vina_docked.pt` | in-repo | 9857 | **8340 (84.6%)** |
| A0 대조 (물리손실 없음) | `novdw` | `novdw` | `results/novdw_gen/novdw_gen_vina_docked.pt` | in-repo | 2054 | 2054 (100%) |
| PIGNet 손실 | `pignet_fixed` | `pignet_fixed_best` | `results/pignet_fixed_best_gen/pignet_fixed_best_vina_docked.pt` | in-repo | 9830 | 7615 (77.5%) |
| PGDiff 손실 | `pgdiff_fixed` | — | `results/pgdiff_fixed_best_gen/pgdiff_fixed_best_vina_docked.pt` | in-repo | 2348 | 1693 (72.1%) |

### 경로 주의사항

- `results/`는 `/lustre/ktori1361/TheSelective_fix/results`로의 **심볼릭 링크**다.
  repo 루트 기준 상대경로로 접근하면 정상 동작한다. git은 심볼릭 링크 너머를 추적하지 못하므로
  이 벤치마크의 SSOT 문서는 `results/`가 아니라 **`analysis/nci_benchmark/`**에 둔다.
  2026-08-05 이전에는 `/scratch2/ktori1361/TheSelective_fix/results`를 가리켰다. `/scratch2`가
  30일간 접근이 없으면 삭제되는 정책이라 `/lustre`로 옮겼고, 파일 개수(217,828)와 바이트 합
  (103,857,589,193)이 양쪽에서 일치함을 확인한 뒤 링크를 바꿨다. **표의 수치는 재측정하지 않았다** —
  같은 파일이 다른 경로에 있을 뿐이다.
- PIDiff는 반드시 `PIDiff_vina_docked_complete.pt`를 쓴다. `_complete`가 없는
  `PIDiff_vina_docked.pt`에는 도킹 점수가 없다.

### 표에 반드시 병기할 경고

- **PIDiff 공식 릴리스는 850분자뿐**이다(포켓당 중앙값 9, 빈 포켓 1개). 다른 모델의 ~9,000과
  한 자릿수 차이라 포켓 단위 평균의 신뢰구간이 훨씬 넓다. `N mols` 컬럼을 반드시 넣고,
  CI 폭 차이를 캡션에 명시한다.
- **IPDiff에는 도킹 정보가 전혀 없다.** 공식 `.pt`는 raw sampler 출력
  (`data`, `pred_ligand_pos`, `pred_ligand_v`, `pred_ligand_pos_traj`, `pred_ligand_v_traj`)이며
  RDKit mol도 vina 필드도 없다. 도킹 성공 분모 규약에 합류시키려면 우리가 직접 도킹해야 한다
  (Phase B). 이 단계를 건너뛰면 IPDiff는 필터에서 전량 탈락하거나 필터가 무음으로 무시된다.

---

## 4. 포켓 정렬 검증 (게이트 통과)

`scripts/build_test_pockets.py` — canonical 순서는 `eval_out/targetdiff/manifest.csv`이며,
`eval_export_sdf.load_grouped`가 flat `.pt`를 정렬해 넣는 기준과 동일하다.
검증은 포켓 디렉토리명이 아니라 **full ligand_filename**으로 한다 (한 포켓 디렉토리에
결정 리간드가 둘 이상 있을 수 있다).

```
[OK] targetdiff       100 pockets ok        [OK] novdw            100 pockets ok
[OK] kgdiff           100 pockets ok        [OK] pignet_fixed     100 pockets ok
[OK] PIDiff_official  100 pockets ok (1 empty slot)
[OK] pidiff_exact     100 pockets ok        [OK] pgdiff_fixed     100 pockets ok
[OK] vina_fixed_best  100 pockets ok        [OK] NATIVE           100 pockets ok
[OK] IPDiff_official  100 pockets ok
GATE PASSED
```

이 게이트가 필요한 이유: 예전에 flat `.pt`를 등장 순서대로 재번호해 **77개 PIDiff 포켓이 엉뚱한
receptor와 짝지어진** 적이 있다. 무음 실패였고, 모든 paired 통계를 오염시켰다.

---

## 5. 도킹 성공 분모 규약

**전 모델·전 지표를 도킹 성공 분자만으로 집계한다.**

근거: 베이스라인 `.pt`는 애초에 도킹 성공 분자만 담고 있다(위 표에서 전부 100%). 우리 것만
실패 분자를 포함하면 우리 모델이 더 넓은 집합에서 채점되어 비교가 불공정해진다.
따라서 `--docked_only`는 **베이스라인에 no-op이고 우리에게만 적용되는** 유일하게 공정한 필터다.

### 실측: `docked` 와 `connected`는 동일한 필터다

8개 세트 전부에서 교차표의 비대각 칸이 **정확히 0**이었다:

| 세트 | 3D 분자 | 도킹 성공 | 조각남 | 도킹성공 ∧ 조각남 |
|---|---|---|---|---|
| targetdiff | 9036 | 9036 | 0 | **0** |
| kgdiff | 8813 | 8813 | 0 | **0** |
| PIDiff 공식 | 850 | 850 | 0 | **0** |
| pidiff_exact | 8213 | 8213 | 0 | **0** |
| novdw | 2054 | 2054 | 0 | **0** |
| vina_fixed_best | 9857 | 8340 | 1517 | **0** |
| pignet_fixed | 9830 | 7615 | 2215 | **0** |
| pgdiff_fixed | 2348 | 1693 | 655 | **0** |

원인: meeko가 다중 조각 리간드를 준비하지 못한다("must have 1 fragment"). 즉 **도킹 실패는
전부 조각남으로 설명된다.** 따라서 별도의 `--connected_only` 필터를 만들지 않고, 이미 검증된
`--docked_only`를 쓴다. 표의 `Docked %` 컬럼은 곧 `Connected %` 컬럼이다.

### 부수 결과 (그 자체로 보고 대상)

물리 손실이 **없는** `novdw`는 조각 분자가 **0개**인데, 접촉 물리를 넣은 세 모델은 전부 조각난다:
vina 15.4%, pignet 22.5%, pgdiff 27.9%. 조각화는 백본이 아니라 **물리 손실이 유발한다**.
NCI 증가를 주장할 때 이 비용을 같은 표에 반드시 병기한다.
