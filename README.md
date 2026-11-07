# 데이터 특성에 따른 CATE 추정 모델의 성능 비교 연구

**연구 질문:** 표본 크기, 처치군 불균형, 처치효과의 복잡성에 따라 CATE 추정 모델(T-Learner, X-Learner, Causal Forest)의 정확도와 안정성은 어떻게 달라지는가?

| 단계 | 내용 | 상태 |
|---|---|---|
| 1 | 프로젝트 구성 + 합성 데이터 생성 파이프라인 | **완료** |
| 2 | 모델 구현 + 예비 실험 1회 (n=1000, p=0.5, linear) | **완료** |
| 3 | 반복 실험 자동화 + 기준 조건 3회 시험 실행 | **완료** |
| 4 | 정식 OFAT 30회 반복 실험 (1차 n=1,000 완료 / 2차 n=10,000 · 3차 n=100,000 예정) | 진행 중 |
| 3 | Criteo Uplift 데이터에서 Qini curve, AUUC, 상위 타기팅 비율별 증분 전환 평가 | 예정 |

## 폴더 구조

```text
cate-benchmark/
├── configs/simulation.yaml      # 기본 실행 조건 + 실험 그리드
├── configs/models.yaml          # 사전 고정한 모델 하이퍼파라미터 + 분할 설정
├── configs/experiment.yaml      # 반복 실험 설계 (기준 조건, 요인 수준, 반복 수, seed)
├── data/synthetic/              # 생성된 CSV + 메타데이터 JSON (Git 제외)
├── data/criteo/                 # Criteo 원본 데이터 위치 (Git 제외)
├── src/data_generation.py       # 데이터 생성 과정(DGP), 진단 통계량, 저장
├── src/models.py                # T-Learner, X-Learner, CausalForestDML 래퍼
├── src/evaluation.py            # 층화 분할, 입력 추출(누수 방지), RMSE/MAE
├── src/simulation.py            # 조건 생성, 반복 1회 실행, 요약 통계
├── scripts/generate_data.py     # 데이터 생성 진입점 (CLI)
├── scripts/run_pilot.py         # 예비 실험 진입점
├── scripts/run_simulation.py    # 반복 실험 진입점
├── tests/test_data_generation.py
├── tests/test_models.py
├── tests/test_simulation.py
└── results/{tables,figures}/    # 분석 결과 (Git 제외)
```

## 환경 설정

Python **3.11 이상**이 필요합니다. 먼저 `python3 --version`(Windows는 `py --version`)으로 버전을 확인하세요.

**macOS / Linux**

```bash
python3 -m venv .venv            # 3.11+ 인터프리터를 명시하려면: python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> conda가 활성화된 상태라면 `python3`가 conda 환경의 인터프리터(3.11 미만일 수 있음)를 가리킬 수 있습니다. 이 경우 `conda deactivate` 후 진행하거나 3.11+ 인터프리터의 전체 경로를 지정하세요.

VS Code에서는 `Python: Select Interpreter`에서 `.venv`를 선택합니다.

## 실행

```bash
python scripts/generate_data.py                       # 기본 조건: n=1000, p=0.5, linear, seed=42
python scripts/generate_data.py --n 10000 --treat-prob 0.1 --effect interaction
pytest                                                # 검증 테스트
```

출력 파일(`data/synthetic/`):

- `synthetic_n{n}_p{p}_{effect}_noise{sd}_seed{seed}.csv` — 열: `X1..X10, T, mu0, true_cate, Y`
- 같은 이름의 `.meta.json` — seed, 전체 생성 조건, 생성식, 난수 추출 순서, 처치/대조군 수, CSV의 SHA-256, 패키지 버전

## 데이터 생성 과정(DGP)

$$
X_1,\dots,X_{10} \overset{iid}{\sim} N(0,1),\quad
T \sim \text{Bernoulli}(p),\quad
Y = \mu_0(X) + T\,\tau(X) + \varepsilon,\quad \varepsilon \sim N(0,\sigma^2)
$$

$$
\mu_0(X) = 0.2X_1 - 0.1X_2 + 0.3X_3
$$

- 처치 배정은 X와 독립인 **무작위 배정**입니다(성향점수 = p로 상수). 이 단계에서는 교란(confounding)을 의도적으로 배제하여, 모델 간 차이가 표본 크기·불균형·효과 복잡성에서만 오도록 했습니다.
- X4~X10은 결과에 영향을 주지 않는 잡음 공변량입니다.

### 처치효과 함수

세 함수는 모두 $\tau(X) = 0.5 + 0.3\,g(X)$ 형태이며, $g$는 $X \sim N(0, I)$에서 **이론적 평균 0, 분산 1**을 갖도록 설계했습니다. 따라서 모든 복잡성 수준에서 $E[\tau]=0.5$, $SD[\tau]=0.3$가 정확히 같습니다. 즉 복잡성을 바꿔도 효과의 평균 크기와 이질성의 크기는 그대로이고 **함수 형태만** 달라집니다.

| effect | $g(X)$ | 선택 이유 |
|---|---|---|
| `linear` | $X_1$ | 기준 조건. 명세의 $0.5 + 0.3X_1$. |
| `nonlinear` | $\dfrac{1}{\sqrt2}\left[\dfrac{X_1^2-1}{\sqrt2} + \dfrac{\sin(\pi X_2)}{s}\right]$, $s=\sqrt{(1-e^{-2\pi^2})/2}$ | 비단조 U자형(이차항)과 주기형(사인) 비선형을 결합한 가법 구조. 두 항 모두 X와의 선형 상관이 거의 0이라 선형 근사로는 포착되지 않음. 각 항을 정확히 표준화한 뒤 $\sqrt2$로 나누어 분산 1을 맞춤. |
| `interaction` | $X_1 X_2$ | 주효과가 없는 순수 상호작용. 독립 표준정규의 곱은 평균 0, 분산 1이므로 추가 보정이 필요 없음. 개별 변수만 보는 모델에는 보이지 않고 변수 간 분할(트리 등)이 필요함. |

$s$의 유도: $Z\sim N(0,1)$에서 $E[\sin(\pi Z)]=0$, $E[\sin^2(\pi Z)] = \tfrac{1}{2}(1 - E[\cos 2\pi Z]) = \tfrac12(1-e^{-2\pi^2})$.

### 스케일 진단 통계량

`generate_data.py`는 같은 X에 세 효과 함수를 모두 적용해 다음 값을 출력합니다: mean, sd, min/q05/median/q95/max, skew, `sd_over_noise`(이질성 SD / 잡음 SD), `linear_r2`(X에 대한 선형회귀가 τ 분산을 설명하는 비율), 이론값.

기준 표본(n=200,000, seed=42) 결과:

| effect | mean | sd | skew | linear_r2 |
|---|---|---|---|---|
| linear | 0.500 | 0.301 | 0.00 | 1.000 |
| nonlinear | 0.501 | 0.301 | 1.01 | 0.001 |
| interaction | 0.500 | 0.301 | 0.01 | 0.000 |

평균과 SD는 같고 선형 설명력만 1에서 0으로 떨어지므로, 복잡성이 스케일과 분리되어 있음을 확인할 수 있습니다.

**해석 시 주의:** 1·2차 적률은 같지만 분포의 모양은 다릅니다. `nonlinear`은 오른쪽으로 치우쳐 있고(skew ≈ 1, $X_1^2$ 항 때문에 τ ≥ 약 0.05), `interaction`은 꼬리가 두껍습니다(극단값이 큼). 결과 해석 시 이 차이도 함께 고려해야 합니다.

### 재현성

- 난수: `numpy.random.default_rng(seed)`(PCG64)에서 **X → T용 균등난수 → ε** 순서로 추출합니다.
- 추출 순서가 고정되어 있어, 같은 seed와 n이면 처치 비율·효과 유형이 달라도 X, 균등난수, ε가 공유됩니다(공통 난수). 조건 간 비교에서 표본 잡음 차이가 줄어들고, `T = 1[U < p]`이므로 p=0.1의 처치군은 p=0.3 처치군의 부분집합이 됩니다.
- 메타데이터의 SHA-256으로 CSV가 변경되지 않았는지 확인할 수 있습니다.

## 실험 조건

| 요인 | 수준 |
|---|---|
| 표본 크기 | 1,000 / 10,000 / 100,000 |
| 처치군 비율 | 0.5 / 0.3 / 0.1 |
| 효과 복잡성 | linear / nonlinear / interaction |
| seed / 잡음 SD | 42 / 1.0 |

전체 27개 조건 그리드는 `configs/simulation.yaml`에 정의만 되어 있으며, 반복 실험은 2단계에서 수행합니다.

## CATE 추정 모델

세 모델은 모두 `fit(X, T, Y)` / `effect(X)` 인터페이스를 따르며, 입력은 **X1~X10, T, Y로만** 제한됩니다(`src/evaluation.py::extract_model_inputs`). `true_cate`와 `mu0`는 테스트 데이터 채점에만 사용합니다.

### 추정 방식의 차이

세 모델은 같은 구조가 아닙니다. T/X-Learner는 **결과 회귀 모델을 조합하는 메타러너**이고, Causal Forest는 **처치효과 자체를 분할 기준으로 삼는 전용 포레스트**입니다.

| | T-Learner | X-Learner | Causal Forest (`econml.dml.CausalForestDML`) |
|---|---|---|---|
| 추정 대상 | 집단별 결과 $\mu_0(x), \mu_1(x)$ | 1단계 $\mu_0, \mu_1$ → 2단계 개인별 효과 대리값 | 잔차화한 Y와 T 사이의 국소 회귀 계수 $\tau(x)$ |
| CATE | $\hat\mu_1(x) - \hat\mu_0(x)$ | $e\,\hat\tau_0(x) + (1-e)\,\hat\tau_1(x)$ | GRF가 직접 추정 |
| 분할 기준 | 결과 Y의 예측 오차 | 결과 Y와 대리 효과 D의 예측 오차 | 처치효과 이질성 (efficacy 기준) |
| 정규화 | 두 모델을 따로 정규화 → 차이에 오차가 쌓이기 쉬움 | 반대 집단 모델로 대리값을 만들어 불균형에 대응 | Honest 분할(분할용·추정용 표본 분리), 교차적합(cross-fitting) |
| 기본 학습기 | `RandomForestRegressor` | `RandomForestRegressor` (4개 모델 모두) | nuisance: `RandomForestRegressor`, 최종: causal forest |

**X-Learner 세부.** 대리 효과는 처치군에서 $D_1 = Y - \hat\mu_0(X)$, 대조군에서 $D_0 = \hat\mu_1(X) - Y$이며, 두 경우 모두 (처치 − 비처치) 방향입니다. 결합 가중치는 Künzel et al. (2019)를 따라 $g(x)=e(x)$입니다. 처치군이 적으면(e가 작으면), 많은 대조군으로 학습한 $\hat\mu_0$를 이용한 $\hat\tau_1$이 더 신뢰할 만하므로 $\hat\tau_1$에 가중치 $1-e$를 줍니다. 무작위 배정이므로 e는 설계값(`treat_prob`)을 사용합니다.

**CausalForestDML 세부.**

- `model_y` = E[Y|X]: `RandomForestRegressor` (T/X-Learner와 같은 하이퍼파라미터)
- `model_t` = P(T=1|X): `DummyClassifier(strategy="prior")`. 무작위 배정이라 성향점수가 상수이므로 각 fold의 처치 비율을 사용합니다. RF로 추정하면 실제로는 없는 X 의존성이 잡음으로 섞이기 때문입니다. `configs/models.yaml`의 `propensity_model: random_forest`로 바꿀 수 있습니다.
- `discrete_treatment=True`, 교차적합 `cv=5`: nuisance 모델을 5-fold out-of-fold 예측으로 잔차화합니다.
- 최종 포레스트: `n_estimators=1000`, `min_samples_leaf=5`, `max_features=1.0`, `honest=True`, `random_state=seed`

### 사전 고정 하이퍼파라미터 (`configs/models.yaml`)

| 항목 | 값 |
|---|---|
| RF 기본 학습기 | `n_estimators=200, min_samples_leaf=5, max_features=1.0, random_state=seed` |
| X-Learner 성향점수 | 설계 처치 비율 (예비 실험: 0.5) |
| Causal Forest | 위 참조 |
| 병렬화 | 모든 모델 `n_jobs=1` (학습 시간을 비교 가능하게) |
| 분할 | 80/20, T로 층화, `random_state=seed` |

## 예비 실험

```bash
python scripts/generate_data.py   # 데이터가 없을 때만
python scripts/run_pilot.py
```

`run_pilot.py`는 저장된 CSV의 SHA-256과 생성 조건을 메타데이터와 대조한 뒤 실행합니다. 결과물:

- `results/tables/pilot_model_comparison.csv` — Model, RMSE, MAE, Train Time (sec)
- `results/tables/pilot_predictions.csv` — 테스트 샘플별 `true_cate`와 세 모델의 예측값
- `results/tables/pilot_experiment_metadata.json` — 데이터 해시, 분할 설정, 집단별 표본 수, 모델 하이퍼파라미터, 라이브러리 버전
- `results/figures/pilot_rmse_comparison.png` — RMSE 막대그래프 (한글 폰트: AppleSDGothicNeo)

비교 기준으로 상수 ATE(학습 데이터의 평균 차이) 예측의 RMSE도 함께 출력합니다. 이질성을 학습한 모델이 이 기준보다 나은지 확인하는 용도입니다.

> 예비 실험은 단 1회 실행이므로 모델 간 우열이나 계산 비용을 결론짓는 근거로 사용하지 않습니다.

## 반복 시뮬레이션 실험

```bash
python scripts/run_simulation.py --only-baseline --reps 3 --tag trial   # 시험 실행
python scripts/run_simulation.py                                        # OFAT 설계, 30회 반복
python scripts/run_simulation.py --design full --tag full               # 27개 조건 완전요인설계
python scripts/run_simulation.py --resume                               # 중단된 실행 이어서
```

**설계 (`configs/experiment.yaml`).** 기본은 OFAT(one-factor-at-a-time) 설계입니다. 기준 조건(n=10,000, p=0.5, linear, σ=1)을 두고 한 번에 한 요인만 바꾸며, 나머지 조건은 고정합니다.

| 실험 | 변화 요인 | 고정 조건 |
|---|---|---|
| sample_size | n = 1,000 / 10,000 / 100,000 | p=0.5, linear |
| treat_ratio | p = 0.5 / 0.3 / 0.1 | n=10,000, linear |
| complexity | linear / nonlinear / interaction | n=10,000, p=0.5 |

세 실험이 기준 조건을 공유하므로 고유 조건은 **7개**이며, 기준 조건은 한 번만 계산합니다(`experiments` 열에 소속 실험을 `|`로 기록). `--design full`은 27개 조건 전체를 실행합니다.

**Seed 규칙.** 반복 r에서

- 데이터 seed = 42 + r: 데이터 생성과 학습·테스트 분할에 사용
- 모델 seed = 1000 + r: 세 모델 모두에 사용

같은 반복 안에서는 세 모델과 상수 ATE 기준선이 **동일한 데이터·분할**을 사용하고, 반복 간에는 seed가 달라 서로 독립적인 데이터가 생성됩니다. 같은 반복의 서로 다른 조건은 같은 데이터 seed를 공유하므로, n이 같으면 X와 잡음이 공통입니다(공통 난수 → 조건 간 짝비교에 유리).

**출력 (`results/tables/`).**

- `<tag>_runs.csv`: 반복 × 모델별 RMSE, MAE, 학습 시간, 예측 평균·표준편차, seed, 집단별 표본 수, 최대 메모리(peak RSS)
- `<tag>_summary.csv`: 조건 × 모델별 반복 수, RMSE 평균·표준편차·95% 신뢰구간(t 분포), MAE, 학습 시간, `precision_note`(계획보다 반복 수가 적거나 신뢰구간 반폭이 평균의 10%를 넘으면 표시)
- `<tag>_paired.csv`: 같은 조건·같은 반복의 모델 쌍별 RMSE 차이(a − b, 음수면 a가 더 정확). 짝비교 t 신뢰구간, p값, 조건 내 Holm 보정 p값, a가 더 나았던 반복 수, `note`(신뢰구간이 0을 포함하는 경우 등)
- `<tag>_metadata.json`: 설계, 조건 목록, seed 규칙, 모델 설정, 환경

결과는 반복이 끝날 때마다 CSV에 추가되므로, 중단되더라도 `--resume`으로 이어서 실행할 수 있습니다. 이어서 실행할 때는 조건 값과 반복 번호로 완료된 반복을 건너뛰고, 기존 결과의 중복·누락·seed 규칙을 먼저 검사합니다. 같은 tag의 결과 파일이 이미 있으면 `--resume` 또는 `--overwrite` 없이는 실행되지 않습니다.

**정식 실험 (tag `formal`).** 3단계로 나누어 실행합니다.

조건 목록과 완료된 반복 수는 `python scripts/run_simulation.py --tag formal --list-conditions`로 확인합니다(파일을 바꾸지 않음). `--condition-id`로 특정 조건만 실행할 수 있고, `--n`·`--only-baseline`과 함께 쓰면 모든 조건을 만족하는 조건만 선택합니다. 다른 조건의 결과가 들어 있는 파일에는 `--overwrite`가 거부됩니다.

`--max-new-reps K`를 주면 아직 완료되지 않은 반복 중 앞의 K개(반복 번호 순)만 실행하고 종료합니다. 같은 명령을 다시 실행하면 다음 K개를 실행하고, 모두 완료된 뒤에는 학습 없이, 파일도 바꾸지 않고 종료합니다. seed는 반복 번호로 정해지므로 나누어 실행해도 한 번에 실행한 결과와 같습니다. 3차 실험(c02, n=100,000)은 이 옵션으로 5회씩 나누어 실행합니다.

```bash
caffeinate -i .venv/bin/python scripts/memory_guard.py --limit-mb 7000 --interval 5 \
  --log results/tables/formal_c02_memory_$(date +%Y%m%d_%H%M).csv -- \
  .venv/bin/python scripts/run_simulation.py --tag formal --condition-id c02 --resume --max-new-reps 5
```

```bash
caffeinate -i .venv/bin/python scripts/run_simulation.py --tag formal --n 1000              # 1차: c00
caffeinate -i .venv/bin/python scripts/run_simulation.py --tag formal --n 10000 --resume    # 2차: c01, c03-c06 한꺼번에
# 또는 조건 하나씩: --condition-id c01 / c03 / c04 / c05 / c06
caffeinate -i .venv/bin/python scripts/memory_guard.py --limit-mb 7000 --interval 5 \
  --log results/tables/formal_n100000_memory_$(date +%Y%m%d_%H%M).csv -- \
  .venv/bin/python scripts/run_simulation.py --tag formal --n 100000 --resume              # 3차: c02
```

> STEP 3 시험 실행(`trial_*`)과 STEP 3-1 점검(`probe_*`)은 기준 조건 n=1,000 설정에서 실행했습니다. 각 파일의 메타데이터에 실제 조건이 기록되어 있습니다. 하이퍼파라미터는 `configs/models.yaml` 값을 그대로 사용합니다.

## 검증 테스트

`tests/test_simulation.py`: OFAT 조건 7개(한 요인만 변화, 기준 조건 공유), 완전요인설계 27개, 데이터·모델 seed 분리, 같은 반복 안에서 모델 간 데이터 공유, 반복 간 독립성, 재현성, 평균·표준편차·신뢰구간 계산.

`tests/test_models.py`: T-Learner의 집단별 모델 학습, X-Learner 대리 효과의 부호와 값, 가중 결합 방향(e=0.1 → τ1 가중치 0.9), 잡음 없는 선형 데이터에서 정확한 복원, Causal Forest 예측의 유효성, 예측 길이, 층화 분할, RMSE/MAE 계산, `true_cate`/`mu0` 변조에 대한 예측 불변성(누수 없음), 동일 seed 재현성.

### `tests/test_data_generation.py`

1. 행 수 = 요청한 n
2. X1~X10, T, Y, true_cate(+mu0) 열 존재
3. 처치군 비율이 p의 4σ 이항 신뢰범위 안에 있음 (n=1,000 / 100,000 × p=0.5/0.3/0.1)
4. 같은 seed → 같은 데이터, 다른 seed → 다른 데이터, CSV 저장 후 다시 읽어도 값 보존
5. 잡음 0일 때 Y = mu0 + T·τ 정확히 일치. 잡음이 있을 때 잔차의 평균≈0, SD≈σ, T·X와 무상관
6. true_cate = 선택한 효과 함수. 세 함수 모두 평균 0.5, SD 0.3이며 linear_r2로 복잡성이 구분됨
7. NaN / ±inf 없음
