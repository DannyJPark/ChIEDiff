# NCI Benchmark — 사전등록 (PREREGISTRATION)

**작성 시각**: 2026-07-24
**커밋 시점**: 신규 태그(IPDiff / PIDiff 공식 / NATIVE) 평가 **이전**, 집계 스크립트 실행 **이전**
**관련 문서**: `analysis/NCI_BENCHMARK_PLAN.md`(원 계획), `analysis/nci_benchmark/provenance.md`(출처),
`analysis/METHODS_interaction_metrics.md`(지표 정의)

---

## 0. 이 문서가 사전등록할 수 있는 것과 없는 것 — 정직한 범위 한정

사전등록은 **아직 보지 않은 결과**에 대해서만 의미가 있다. 이 프로젝트는 일부 데이터가
이미 존재하므로, 그 부분을 사전등록이라고 부르면 거짓이 된다. 범위를 명시적으로 나눈다.

### 진짜 사전등록 (결과를 아직 보지 못함)

| 항목 | 상태 |
|---|---|
| **IPDiff** 상호작용 지표 | 아직 재구성·도킹·평가 전. 어떤 NCI 값도 보지 않았다. |
| **PIDiff 공식 릴리스** PoseCheck/PLIP 지표 | `eval_out`/`eval_plip` 태그 자체가 없다. |
| **IRDiff** 전체 | 생성조차 하지 않았다. |
| **A2/A3 ablation** (hbond만 / hydrophobic만) | 재학습 필요. 미실행. |

### 사전등록이 **아닌** 것 (데이터가 이미 있음 — 정직하게 사후분석으로 표기)

TargetDiff / KGDiff / pidiff_exact / novdw / pignet_fixed / pgdiff_fixed / **Ours(vina_fixed_best)**의
PLIP 상호작용 값은 이미 산출되어 `analysis/RESULTS_interactions_2026-07.md`에 실려 있다.
이들 사이의 비교는 **exploratory(사후)** 로 표기하며, 논문에서 사전등록된 검정인 척하지 않는다.

이 구분을 흐리지 않는 것이 이 문서의 첫 번째 목적이다.

---

## 1. 검증할 주장

> Vina 스코어링 함수의 hydrophobic 항과 hbond 항을 diffusion 학습 손실에 포함시키면,
> 생성 분자가 **독립 계측기로 실제 측정되는** 비공유 상호작용(NCI)을 더 많이 형성한다.

"독립 계측기"가 핵심이다. Vina 항을 최소화하도록 훈련한 모델을 Vina로 채점하면 순환논증이므로,
채점은 PoseCheck(ProLIF)와 PLIP으로 한다. 둘 다 Vina에 없는 기준을 쓴다.

---

## 2. 사전 예측 (pre-registered prediction)

Vina의 두 항은 표면거리 `d = r_ij - R_i - R_j`에 대해 정의된다.

```
h_hbond(d)  = 1  (d < -0.7);  -d/0.7  (-0.7 <= d < 0);  0  (d >= 0)
h_phobic(d) = 1  (d < 0.5);   1.5-d   (0.5 <= d < 1.5);  0  (d >= 1.5)
```

Vina 반경(C 1.9, N 1.8, O 1.7 Å)을 대입하고 PLIP 3.0.0의 실제 임계값과 대조하면:

| 항 | Vina 최적 영역 | PLIP 기준 | 정합성 |
|---|---|---|---|
| hydrophobic | C–C ≲ 4.3 Å | C–C < 4.0 Å | **정합** → 전이 예상 |
| hbond | N/O–O ≲ 2.8 Å | d(D,A) < 4.1 Å **AND** ∠(D–H⋯A) > 100° | **불완전** — Vina에 각도 항이 없음 |

### 예측 P1 (주 예측)
`Hydrophobic / heavy atom`이 Ours에서 A0 대조(`novdw`) 대비 **유의하게 증가**한다.
(포켓 단위 paired Wilcoxon, Holm 보정 후 p < 0.05, 효과크기 병기)

### 예측 P2
`HBDonor+HBAcceptor / heavy atom`의 증가폭은 **P1보다 작거나 유의하지 않을 수 있다.**

**P2가 기각되지 않아도 실패가 아니다.** 그 경우 *"거리 기반 guidance는 방향성 상호작용을
유도하지 못한다"* 는 물리적 발견으로 보고한다. 이 문장을 결과를 보기 전에 못박아 둔다 —
사후에 "원래 그럴 줄 알았다"로 재서술하는 것을 막기 위해서다.

### 예측 P3 (비용)
NCI 증가에는 대가가 따른다. Ours는 A0 대비 **조각률과 clash가 악화**될 것으로 본다.
근거: 이미 측정된 조각률(novdw 0% vs vina 15.4% / pignet 22.5% / pgdiff 27.9%).
→ P1이 성립하더라도 P3가 함께 성립하면 결론은 "개선"이 아니라 **"트레이드오프"** 로 서술한다.

### 예측 P4 (도구 일치)
PoseCheck와 PLIP의 **모델 순위가 일치**할 것으로 본다.
불일치 시 은폐하지 않고 그 사실 자체를 discussion에 명시한다(원 계획 §5.2, §9).

---

## 3. 집계 규칙 — 결과를 보기 전에 확정

데이터를 본 뒤 규칙을 바꾸는 것(garden of forking paths)을 막기 위해 아래를 고정한다.

### 3.1 분모: 도킹 성공 분자만 (전 모델 공통)

베이스라인 `.pt`는 애초에 도킹 성공 분자만 담고 있다(`provenance.md` §3에서 전부 100% 확인).
우리 것만 실패 분자를 포함하면 더 넓은 집합에서 채점되어 불공정하다.
→ **전 모델·전 지표를 `--docked_only`로 집계한다.** 베이스라인에는 no-op이다.

실측 결과 `docked ⟺ connected`가 정확히 성립하므로(8개 세트 전부 비대각 0), 이 규칙은
"조각나지 않은 분자만"과 동일하다. 별도 필터를 만들지 않는다.

**은폐 방지 조건 (필수)**
- 메인 표에 `Docked %` 컬럼을 넣는다 — 모델별 제외율(우리만 84.6%)이 곧 보이게.
- 필터 없는 전체 분자 표를 부록에 병기한다.
- 조각률 자체를 **모델 품질 결과로** 별도 보고한다(`fragmentation_census.py`).
  "우리 모델은 15.4%가 조각난다"는 숨길 사실이 아니라 보고할 사실이다.

### 3.2 분석 단위: 포켓

포켓당 ~100분자는 독립 표본이 아니다(같은 단백질·같은 조건). 분자를 풀링해 검정하면
pseudo-replication이 되어 p값이 과대평가된다.
1. 포켓 **내부**에서 먼저 평균 → 포켓당 값 1개
2. 포켓 **간** paired Wilcoxon signed-rank (n=100, 전 모델이 같은 포켓을 공유하므로 paired)
3. 포켓 단위 bootstrap 95% CI (B=10,000)
4. `n`(포켓 수)을 항상 병기

### 3.3 주 지표 (이것만 confirmatory, 나머지는 exploratory)

- `Hydrophobic / heavy atom` (PoseCheck)
- `(HBDonor + HBAcceptor) / heavy atom` (PoseCheck)
- `Clash` (PoseCheck)
- `Strain energy` — **중앙값** 사용 (long tail)

raw count 단독 보고 금지. 반드시 heavy-atom 정규화 값과 원자 수를 병기한다.
NCI 증가를 clash 없이 보고하지 않는다.

### 3.4 다중비교

4개 비교 × 4개 지표에 Holm–Bonferroni 보정. p값만 쓰지 않고 효과크기(rank-biserial)와 CI 병기.

### 3.5 원자 수 통제 — 강제 불가, 정규화로 대체

원 계획 §3.3은 `(pocket, sample_idx) → N_M` 테이블을 전 모델에 주입하라고 했으나,
IPDiff·TargetDiff·KGDiff·PIDiff는 **이미 생성된 외부 샘플**이라 소급 통제가 불가능하다.
→ 계획서 자체의 fallback을 채택한다: **heavy-atom 정규화를 주 지표로 삼고, 원자 수 분포를
부록에 첨부**한다. 모델 간 평균 원자 수 차이가 1.0을 넘으면 보고한다(§4).

---

## 4. 중단·보고 의무

아래 상황에서는 임의 판단으로 진행하지 않고 즉시 중단·보고한다.

- 포켓 정렬 게이트 실패 (`build_test_pockets.py`) — **2026-07-24 통과함**
- IPDiff 도킹 커버리지가 다른 모델 대비 극단적으로 낮음 (< 70%)
- IPDiff에서 `docked ⟺ connected`가 성립하지 않음 (다른 8개 세트에서는 성립)
- 모델 간 평균 원자 수 차이 > 1.0
- 어떤 모델의 sanitize 실패율 > 5%
- **PoseCheck와 PLIP의 모델 순위가 뒤바뀜** (P4 기각)
- `--docked_only` on/off에서 베이스라인 행 값이 움직임 (필터 no-op이어야 정상 → 조인 결함)
- IRDiff retrieval pool을 재구현해야 하는 상황

---

## 5. 서명

이 문서는 위 "진짜 사전등록" 항목의 데이터를 산출하기 **전에** 커밋된다.
커밋 해시가 시점의 증거이며, 이후 예측을 수정할 경우 **원문을 삭제하지 않고 개정 이력을 덧붙인다.**
