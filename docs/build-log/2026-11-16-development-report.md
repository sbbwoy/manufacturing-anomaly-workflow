# 제조 공정 이상탐지 워크플로우 개발 리포트

정리일은 2026년 11월 16일이다. Manufacturing Anomaly Detection Workflow의 기초 EDA부터 최종 평가까지의 작업 과정과 의사결정을 정리했다. 사용자가 작성한 리포트를 바탕으로 저장된 산출물과 Git 기록을 대조했다.

데이터는 UCI AI4I 2020 합성 데이터다. Python, pandas, NumPy, scikit-learn, matplotlib, seaborn을 사용했다. [프로젝트 목표](../../GOALS.md) · [개발 계획](../../PLANS.md) · [데이터 출처](https://doi.org/10.24432/C5HS5C)

## 1. 작업 개요

제조 공정의 고장 징후를 탐지하고, 데이터 준비·라벨 정의·모델 비교·평가를 추적 가능한 워크플로우로 구성했다. 기초 EDA를 출발점으로 변수 조합 분석, 라벨 감사, 전처리, 기준 모델 학습, Validation 선택과 Test 최종 평가를 진행했다.

분석 대상은 10,000건이며 원본 Machine failure 양성은 339건(3.39%)이다. 원본을 보존하고 UID, Product ID, 원본 행 위치, 분할명과 라벨 불일치 여부로 결과를 추적했다. 전처리 및 평가 산출물과 원본의 해시를 실행 기록에 남겼다. [기초 EDA](../../artifacts/eda/report.md) · [전처리 실행 요약](../../artifacts/preprocessing/summary.json)

## 2. 작업 과정과 의사결정 근거

### 공정 변수의 조합 탐색

회전 속도와 토크의 Pearson 상관계수는 약 −0.875였다. HDF=1인 115건의 공정 온도−공기 온도 중앙값은 8.3 K로, HDF=0의 9.8 K와 차이가 있었다. 제품 유형별 OSF 비율은 L 1.45%, M 0.30%, H 0.20%였다. H형 OSF는 2건이어서 해석 범위가 작다. [추가 EDA 보고서](../../artifacts/eda_interactions/report.md)

이 관찰을 바탕으로 온도 차이, 기계적 동력, 토크×공구 마모를 행별 파생변수로 구성했다. 변수 조합의 물리적 해석 가능성을 고려한 설계이며, 파생변수 추가의 성능 효과를 분리하는 제거 비교 실험은 수행하지 않았다. 관측된 상관과 그룹 차이로 고장 원인을 단정하지 않았다. [전처리 정책](../../PREPROCESSING.md)

### 라벨 정합성 감사와 원본 보존

| 항목 | 확인 결과 |
|---|---:|
| 전체 관측 | 10,000건 |
| Machine failure=1 | 339건 |
| 고장 플래그 5개의 OR=1 | 348건 |
| 라벨 불일치 | 27건 |
| 정상 라벨이지만 플래그 존재 | 18건 |
| 고장 라벨이지만 플래그 없음 | 9건 |

공식 설명과 CSV의 빈도는 TWF 51/46건, RNF 5/19건으로 달랐다. HDF를 저장된 소수값에 대해 엄격히 재계산하면 103건으로, 플래그 115건과 12건 차이가 났다. 이 12건은 온도 차이가 정확히 8.6 K인 경계 행이다. PWF와 OSF는 검사한 규칙과 모두 일치했다. [라벨 검증 보고서](../../artifacts/validation/report.md) · [불일치 실제 값](../../artifacts/validation/label_disagreements.csv)

불일치 원인과 의도된 정답을 확정할 근거가 부족해 원본 Machine failure를 기준 라벨로 유지했다. 27건을 삭제·수정하지 않고 감사 정보로 남겼다. UID와 Product ID는 추적에 사용하고, 고장 유형 플래그와 그 파생값은 예측 입력에서 제외했다. [검증 요약](../../artifacts/validation/summary.json)

### 재사용 가능한 전처리

고장 희소성을 고려해 원본 라벨로 계층 분할하고 random_state=42를 고정했다.

| 집합 | 관측 | 고장 | 라벨 불일치 |
|---|---:|---:|---:|
| Train | 6,000 | 203 | 15 |
| Validation | 2,000 | 68 | 6 |
| Test | 2,000 | 68 | 6 |

입력은 Type과 수치 공정 변수 5개다. Type의 One-Hot Encoding 3열, 수치 원변수 5열과 파생변수 3열을 합해 11개 특징을 생성했다. Pipeline과 ColumnTransformer를 사용하고, StandardScaler의 통계량은 Train에서만 추정했다. 결측값·미지 Type·잘못된 스키마는 오류로 처리하며 전처리 객체를 저장했다. [전처리 정책](../../PREPROCESSING.md) · [실행 요약](../../artifacts/preprocessing/summary.json)

### 지도학습과 비지도 기준 모델

Logistic Regression은 class_weight=balanced, max_iter=3000, random_state=42로 원본 고장 라벨을 학습했다. Isolation Forest는 n_estimators=200, contamination=auto, random_state=42로 Train 특징만 학습했다. 두 모델은 같은 전처리와 분할을 사용한다. Logistic Regression은 고장 클래스 점수, Isolation Forest는 부호를 반전한 decision_function으로 큰 값이 더 이상한 관측을 나타내도록 저장했다. [모델링 설계](../../MODELING.md) · [모델링 실행 기록](../../artifacts/modeling/summary.json)

원본 고장 라벨에 대한 순위 성능은 AP와 ROC-AUC로 비교했다. AP는 비보간 평균 정밀도이며 사다리꼴 PR 면적과 구분한다. 임계값은 Validation F2를 최대화하고 동률이면 높은 값을 선택했다. F2는 Recall을 더 중요하게 반영하는 잠정 기준이며, 실제 미탐·오탐 비용을 측정해 정한 기준은 아니다. [Validation 평가 보고서](../../artifacts/evaluation/report.md)

## 3. 모델 성능과 변화

### Validation 비교

| 지표 | Logistic Regression | Isolation Forest |
|---|---:|---:|
| AP | 0.421178 | 0.154999 |
| ROC-AUC | 0.918821 | 0.807659 |
| 임계값 | 0.7218245828133553 | 0.054546713119515866 |
| Precision | 0.284884 | 0.153409 |
| Recall | 0.720588 | 0.397059 |
| F1 | 0.408333 | 0.221311 |
| F2 | 0.551802 | 0.301339 |
| TN / FP / FN / TP | 1809 / 123 / 19 / 49 | 1783 / 149 / 41 / 27 |

이 Validation에서는 Logistic Regression의 AP가 더 높았다. 이를 모델 선택 근거로 삼고 F2 최대화 임계값 0.7218245828133553을 최종 평가에 고정했다. 같은 Validation에서 임계값을 선택하고 성능을 측정했으므로 선택 편향이 남는다. [원본 평가 수치](../../artifacts/evaluation/metrics.csv) · [선택 설정](../../config/final_holdout.json)

불일치 6건을 제외하면 F2는 Logistic Regression 0.568445, Isolation Forest 0.309633이었다. 플래그 OR을 대체 라벨로 사용하면 각각 0.563063과 0.301339였다. 선택 임계값은 시나리오 간 고정했으며 모델의 순위는 유지됐다. 대체 라벨이 정답이라는 근거는 확정되지 않았다. [라벨 민감도 분석](../../artifacts/evaluation/sensitivity.csv)

### 고정 모델의 최종 Test 평가

PR #2가 main에 병합된 것을 확인한 뒤, Test 접근 전에 모델·임계값·선택 근거·모델 파일 해시를 설정 파일에 기록했다. 저장된 Logistic Regression으로 Test 2,000건에 대해 predict_proba를 1회 호출했다. 모델 재학습, 임계값 재조정, 재선택을 수행하지 않았다. [고정 설정](../../config/final_holdout.json) · [최종 실행 기록](../../artifacts/final_holdout_execution.json)

| 지표 | Validation | Test | Test − Validation |
|---|---:|---:|---:|
| AP | 0.421178 | 0.412090 | −0.009088 |
| ROC-AUC | 0.918821 | 0.925428 | +0.006607 |
| Precision | 28.49% | 32.82% | +4.34%p |
| Recall | 72.06% | 63.24% | −8.82%p |
| F1 | 0.408333 | 0.432161 | +0.023827 |
| F2 | 0.551802 | 0.533499 | −0.018303 |
| FP | 123 | 88 | −35 |
| FN | 19 | 25 | +6 |

차이는 반올림 전 원본 수치로 계산했다. 동일 모델과 임계값을 서로 다른 표본에 적용한 비교이며, 추가 학습에 따른 성능 변화로 해석할 수 없다. AP는 감소하고 ROC-AUC는 상승했다. 이 차이의 통계적 유의성이나 동등성을 검증한 실험은 수행하지 않았다. [Validation 수치](../../artifacts/evaluation/metrics.csv) · [Test 수치](../../artifacts/final_holdout/metrics.csv)

| Test 실제 라벨 | 정상 예측 | 고장 예측 |
|---|---:|---:|
| 정상 | 1,844 (TN) | 88 (FP) |
| 고장 | 25 (FN) | 43 (TP) |

실제 고장 68건 중 43건을 탐지했고, 고장으로 분류한 131건 중 실제 고장은 43건이었다. Precision은 상승했지만 Recall은 하락했고, Recall에 더 큰 가중치를 둔 F2도 감소했다. UID로 각 오탐·미탐을 추적할 수 있다. [혼동행렬](../../artifacts/final_holdout/confusion_matrix.csv) · [UID별 오류](../../artifacts/final_holdout/test_errors.csv) · [최종 보고서](../../artifacts/final_holdout/report.md)

## 4. 검증과 재현성

- 원본 CSV와 기존 산출물의 SHA-256을 실행 전후 대조했다. 고정된 모델과 설정의 해시를 검사하고, Test 추적정보를 분할 manifest와 대조했다. 수치 특징과 UID의 대응은 기존 전처리의 생성 순서에 의존한다. [최종 실행 요약](../../artifacts/final_holdout/summary.json)
- 마지막 개발 단계의 저장된 검증 기록은 신규 최종 평가 테스트 4개와 기존 18개, 총 22개 통과를 나타낸다. 합성 입력으로 스키마, 재현성, Train 전용 fit, 추적정보, 누수 차단, 단일 추론과 중복 실행 방지를 검증했다. 이 문서 작성에서는 평가나 테스트를 다시 실행하지 않았다. [검증 기록](../../artifacts/final_holdout/verification.json)
- 최종 실행 기록은 배타적으로 생성하며 실패해도 남긴다. 같은 저장소에서 자동 재실행을 차단하지만 기록의 수동 삭제나 다른 사본에서의 실행까지 막는 전역 제어는 아니다. [최종 평가 구현](../../evaluate_holdout.py)
- 직접 의존성 6개와 필수 간접 의존성 13개의 설치 버전을 고정했다. 기록된 검증 환경은 Windows CPython 3.12.14이며 새 환경 설치 검증은 미수행이다. [requirements.txt](../../requirements.txt) · [전처리 정책](../../PREPROCESSING.md)

## 5. 한계와 후속 과제

1. 합성 데이터의 결과이므로 실제 생산 설비의 운영 성능을 입증하지 않는다.
2. 라벨 불일치 27건과 일부 고장 생성 규칙의 정밀도·버전 차이는 미확인이다.
3. 초기 전체 데이터 EDA에 Test가 노출됐다. 시간·설비·로트 독립성도 확인하지 못했다.
4. F2 기준은 실제 운영 비용을 반영하지 않는다. 클래스 가중치를 사용한 Logistic Regression 점수가 보정된 고장 확률인지도 검증하지 않았다.
5. 신규 무라벨 CSV의 검증·변환·추론·저장을 연결하는 통합 배치 CLI는 예정 단계다. EDA 그래프는 구현했으며, 운영 탐지 결과를 통합해 검토하는 시각화는 후속 범위다.

다음 작업은 고정된 전처리 객체와 모델을 재사용하는 신규 데이터 배치 추론과 결과 검토 기능이다. 이번 Test 결과를 근거로 모델·임계값을 다시 선택하지 않는다. 운영 비용, 예측 시점의 변수 가용성, 새로운 독립 데이터에서의 검증 기준을 구체화할 필요가 있다. [개발 계획](../../PLANS.md)

## 6. 회고

라벨의 의미와 불일치를 먼저 확인하면서 이후 학습의 기준을 명확히 할 수 있었다. 불확실한 행을 감사 정보로 보존한 결정은 라벨 민감도를 비교하고 결과를 원래 UID까지 추적하는 데 연결됐다.

ROC-AUC와 운영 임계값의 성능을 함께 보아야 한다는 점도 확인했다. Test ROC-AUC가 약 0.9254여도 고장 25건을 놓쳤다. 운영 적용 판단에는 점검 여력과 미탐·오탐 비용, 외부 검증이 더 필요하다.

전처리 fit 범위, 입력 해시, 고정 설정과 실행 기록을 명시하면서 반복 분석의 조건을 구체화했다. 현재 확보한 모델 개발·평가 구성요소를 통합 배치 실행으로 연결하는 작업이 남아 있다.

## 7. 주요 변경 이력

| 작업 | 커밋 |
|---|---|
| 추가 공정 변수 EDA | [e284a00](https://github.com/sbbwoy/manufacturing-anomaly-workflow/commit/e284a00) |
| 라벨 정합성 감사 | [0df04eb](https://github.com/sbbwoy/manufacturing-anomaly-workflow/commit/0df04eb), [02e678f](https://github.com/sbbwoy/manufacturing-anomaly-workflow/commit/02e678f) |
| 전처리 Pipeline | [ee4cbfa](https://github.com/sbbwoy/manufacturing-anomaly-workflow/commit/ee4cbfa) |
| 기준 모델 학습 | [9ea0a41](https://github.com/sbbwoy/manufacturing-anomaly-workflow/commit/9ea0a41) |
| 실제 기준 모델 산출물 | [d351665](https://github.com/sbbwoy/manufacturing-anomaly-workflow/commit/d351665) |
| Validation 평가 | [e842ba8](https://github.com/sbbwoy/manufacturing-anomaly-workflow/commit/e842ba8), [PR #2](https://github.com/sbbwoy/manufacturing-anomaly-workflow/pull/2) |
| 최종 Test 평가 | [453b8ee](https://github.com/sbbwoy/manufacturing-anomaly-workflow/commit/453b8ee), [PR #3](https://github.com/sbbwoy/manufacturing-anomaly-workflow/pull/3) |
