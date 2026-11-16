# 단계별 개발 계획

현재 저장소에는 예지보전 관련 참고 노트북이 있다. 아래 체크박스는 워크플로우 개발 단계의 완료 여부를 나타낸다. 완료 시 체크박스를 갱신하고 산출물 경로와 검증 결과를 기록한다.

## 1. 데이터 탐색

- [x] 공정 변수 조합 추가 EDA를 `eda_interactions.py`로 실행했다. [추가 보고서](artifacts/eda_interactions/report.md), 그래프 4개와 통계표 5개를 별도로 저장했다. 원본 CSV와 기존 EDA 산출물의 해시 보존을 확인했다. HDF는 115건, 제품 유형별 OSF는 L 87건, M 9건, H 2건이다.
- 추가 분석 재실행은 `python eda_interactions.py --output artifacts/eda_interactions_new`로 수행한다. 기존 출력 폴더가 있으면 덮어쓰기를 거부한다.

- [x] EDA 입력으로 [UCI AI4I 2020 합성 데이터](https://doi.org/10.24432/C5HS5C)를 확보했다. 원본은 `data/ai4i2020.csv`에 저장했다.
- [x] [EDA 스크립트](eda.py)를 원본 CSV로 실행해 [분석 보고서](artifacts/eda/report.md), CSV 통계표 11개, PNG 그래프 5개를 생성했다. 10,000행 중 고장은 339건(3.39%)이며, 실행 전후 입력 파일의 SHA-256이 일치했다.
- 재실행 명령은 `python eda.py --input data/ai4i2020.csv --output artifacts/eda`이다. 실행 환경에 pandas, matplotlib, seaborn이 필요하다.
- [x] `validate_labels.py`로 라벨 정합성을 검증하고 [검증 보고서](artifacts/validation/report.md)에 27건의 UID·실제 값, 공식 빈도 비교, 규칙 검증 및 라벨 처리 방안을 저장했다. PWF/OSF 규칙은 전부 일치하고, HDF의 엄격한 소수 비교는 경계 12건에서 달랐다. 원본과 기존 EDA의 해시 보존을 확인했다.
- [ ] 공식 설명과 CSV의 불일치 원인 및 생성 코드·원래 수치 정밀도를 확인한다. 관측 순서의 의미와 설비·로트 구분 가능성도 미확인이다. 기준 학습 라벨은 사용자 지정에 따라 원본 Machine failure로 결정했다.
- [ ] 데이터 후보의 출처, 사용 조건, 스키마, 관측 단위, 결측·중복, 라벨 유무와 시간·설비별 분포를 확인한다.
- **완료 조건** — 분석용 데이터셋과 선정 근거, 데이터 품질 이슈, 이상 정의와 라벨의 한계를 문서화했다. 데이터셋은 미정이다.

## 2. 전처리

- [x] [전처리 정책](PREPROCESSING.md)과 [Pipeline](preprocess.py)을 구현했다. 시드 42의 계층 분할은 Train/Validation/Test 6000/2000/2000행이며, 고장은 203/68/68건이다. 인코딩과 스케일링의 fit은 Train에만 실행했다.
- [x] 불일치 27건을 유지하고 기존 감사 함수를 재사용해 추적 정보를 저장했다. 각 split의 불일치는 15/6/6건이다. 결과는 artifacts/preprocessing/에 저장했고 원본과 기존 산출물의 해시 일치를 확인했다.
- [x] 신규 전처리 테스트 7개와 기존 라벨 감사 테스트 4개, 총 11개가 통과했다. 합성 입력으로 스키마·결측·미지 Type·분할 재현성·누수 차단·저장 객체 복원과 행 정렬을 검증했다.
- [ ] 시간·설비·로트 독립성과 예측 시점의 변수 가용성을 검증한다. 기존 전체 데이터 EDA로 인해 Test가 탐색에 노출된 한계가 있다.
- **완료 조건** — 정상 입력과 경계 사례에서 전처리 동작을 검증했다. 분할 간 시간·설비·로트 누수 가능성을 점검하고 학습 데이터에서 추정한 전처리 설정을 재사용할 수 있다.

## 3. 이상탐지 모델링

- [x] [모델링 설계](MODELING.md)와 [학습 스크립트](train_models.py)를 추가했다. Logistic Regression(지도학습)과 Isolation Forest(비지도)로 같은 Train/Validation 전처리 입력을 사용하고, 두 모델의 점수와 UID 추적 정보를 저장한다.
- [x] 소규모 합성 입력 기반 신규 단위 테스트 3개 통과: 재현성, 결과 정렬, 잘못된 입력/매니페스트 차단, 비지도 학습에서 라벨 미사용, Test 미접근.
- [ ] 실제 AI4I 10,000건으로 `python train_models.py --input artifacts/preprocessing --output artifacts/modeling`을 실행하고 모델 파일·점수 파일·실행 기록을 검증한다. 기존 전체 테스트 재실행도 필요하다.
- [x] 기존 Validation 점수로 두 모델의 평가 지표와 F2 최대화 임계값을 결정했다. 상세 결과는 4단계와 artifacts/evaluation/report.md에 기록했다.
- **완료 조건** — 기준 방법과 후보의 실행 설정을 기록했고 이상 점수를 원래 관측값과 연결할 수 있다. 최종 Test 평가 모델은 Validation AP가 가장 높은 Logistic Regression으로 고정했다.

## 4. 성능 평가

- [x] 최신 main과 동일한 원격 feat/ai4i-validation-evaluation 브랜치에서 기존 validation_scores.csv를 재사용했다. Validation 2,000행/고장 68건의 AP(비보간 평균 정밀도), ROC-AUC 및 F2 최대화 임계값을 산출했다.
- [x] Logistic Regression은 AP 0.421178, ROC-AUC 0.918821, 임계값 0.7218245828133553, F2 0.551802, FP/FN 123/19였다. Isolation Forest는 AP 0.154999, ROC-AUC 0.807659, 임계값 0.054546713119515866, F2 0.301339, FP/FN 149/41이었다.
- [x] [평가 코드](evaluate_validation.py), [결과 보고서](artifacts/evaluation/report.md), 지표·혼동행렬, UID별 FP/FN 및 라벨 불일치 6건의 민감도 분석을 저장했다. 민감도 분석은 선택된 임계값을 고정하고 원본/불일치 제외/플래그 OR 대체 라벨을 비교했다.
- [x] 신규 합성 데이터 테스트 4개와 기존 14개, 총 18개가 통과했다. 전체 테스트는 임시 작업 디렉터리에서 실행해 기존 전처리 테스트의 산출물 해시 검사도 실제 Test 파일을 읽지 않게 했다. 실제 평가의 파일 접근 감시 결과는 artifacts/evaluation/access_audit.json에 저장했다.
- [x] PR #2의 main 병합(5ed2fb9)을 확인하고 최신 main에서 작업했다. Test 접근 전에 config/final_holdout.json에 Logistic Regression, 임계값 0.7218245828133553, Validation AP 최고·F2 최대화 선택 근거와 모델 파일 해시를 고정했다.
- [x] [최종 평가 코드](evaluate_holdout.py)는 기존 load_split과 metrics를 재사용한다. Test 2,000행/고장 68건의 저장 모델 추론을 1회 수행했다. 재학습·임계값 조정·모델 재선택은 수행하지 않았다.
- [x] Test AP 0.412090, ROC-AUC 0.925428, Precision 0.328244, Recall 0.632353, F1 0.432161, F2 0.533499였다. TN/FP/FN/TP는 1844/88/25/43이며 [보고서](artifacts/final_holdout/report.md), UID별 예측·오류, 혼동행렬과 입력 해시를 저장했다.
- [x] 원본과 기존 산출물의 해시 보존, manifest 추적정보와 중복 검사, 불일치 6건 보존을 확인했다. artifacts/final_holdout_execution.json의 배타적 생성으로 중복 실행을 막으며 실패 시에도 기록을 보존한다.
- [x] 최종 평가 합성 데이터 테스트 4개와 기존 18개, 전체 22개가 통과했다. 테스트는 임시 작업 디렉터리에서 실행해 실제 Test를 읽지 않았다. 고정 설정·모델 해시, 단일 추론, fit 차단, 입력 오류·추적 불일치, 성공·실패·기존 claim 이후 재실행 차단을 검증했다.
- 한계 — 라벨 불일치 원인, 시간·설비 독립성은 미확인이고 과거 전체 데이터 EDA에 Test가 노출됐다. 최종 결과로 모델이나 임계값을 재조정하지 않는다.
- **완료 조건** — Validation 비교와 선택 근거, Test 최종 평가 및 UID별 오탐·미탐과 한계를 기록했다. 목표 성능 수치는 미정이다.

## 5. 자동화

- [x] docs/build-log/2026-11-16-development-report.md에 작업 과정·의사결정·Validation/Test 성능 변화·한계·회고를 정리하고 README에 링크를 추가했다. 저장된 평가 수치와 검증 기록을 대조했으며 분석과 최종 Test 평가는 재실행하지 않았다.

- [x] 실제 Python import와 설치 메타데이터를 대조해 requirements.txt를 생성했다. 직접 의존성 6개와 필수 간접 의존성 13개의 설치 버전을 고정했다. Windows CPython 3.12.14 환경에서 Python 버전·패키지 의존성 조건 충돌이 없고, 기존 테스트 14개가 통과했다. 새 환경 설치 검증은 미수행이다.

- [ ] 입력 검증부터 전처리, 탐지, 결과 저장까지 실행할 진입점을 만들고 설정과 오류를 기록한다. 실행 방식과 저장 형식은 추후 결정한다.
- **완료 조건** — 문서화한 설정으로 전체 흐름을 실행하고 결과를 재현했다. 잘못된 입력으로 실패한 실행이 정상 산출물로 처리되지 않음을 확인했다.

## 6. 시각화

- [ ] 관측값과 이상 점수, 임계값, 이상 후보를 함께 살펴볼 시각화를 만든다. 출력 형식과 도구는 추후 결정한다.
- **완료 조건** — 시각화의 이상 후보를 저장된 탐지 결과와 대조했고, 축·단위·범례로 결과를 해석할 수 있다.
