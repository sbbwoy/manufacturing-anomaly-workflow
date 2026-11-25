# AI4I 모델링: 기준 모델과 이상 점수

## 실행

`python train_models.py --input artifacts/preprocessing --output artifacts/modeling`

`python compare_unsupervised.py --input artifacts/preprocessing --output artifacts/unsupervised_comparison`

`python -m unittest discover -s tests -p "test_*.py"`

출력 폴더가 존재하면 덮어쓰지 않고 실패한다. 추가 실험은 새로운 `--output`을 사용한다.

## 모델과 점수

- Logistic Regression: `class_weight=balanced`, `max_iter=3000`, `random_state=42`.
  라벨 `Machine failure`로 Train 6,000건에 학습한다. 점수는 고장 클래스의 예측 확률이다.
  클래스 가중치를 적용했으므로 확률 보정(calibration) 여부는 후속 평가에서 검토한다.
- Isolation Forest: `n_estimators=200`, `contamination=auto`, `random_state=42`.
  Train의 **특징만** 사용해 비지도 학습한다. 점수는 `-decision_function`이며 클수록 이상에 가깝다.
  이상 점수는 실제 고장 발생 확률과 다르다.
- 두 모델 모두 동일한 **기존 11개 전처리 특징과 Train/Validation 분할**을 사용한다.
  이 단계에서 성능 기준, 임계값, 하이퍼파라미터 탐색은 수행하지 않는다.
- Test 입력과 라벨은 읽거나 점수화하지 않는다. 검증 기반 의사결정 후 별도 평가 단계에서 한 번 사용한다.

## 출력

- `train_scores.csv`, `validation_scores.csv`: UID, Product ID, 원본 라벨,
  원본 행, 라벨 불일치 여부, 분할명, 두 모델의 점수
- `logistic_regression.joblib`, `isolation_forest.joblib`: 학습된 모델
- `summary.json`: 설정, 입력 해시, 점수 정의, 분할 정보 및 제약

## 검증 및 한계

- 입력 특징 11개를 명시적으로 허용하므로 ID·고장 플래그·라벨을 특징에 포함하지 않는다.
- 원본 `Machine failure`는 27건의 불일치 관측을 포함한다. 삭제·수정 없이 기록한다.
- Preprocessing의 `tracking.csv`와 `split_manifest.csv`의 행 순서·UID·라벨을 검증한다.
  특징값 CSV의 행 정렬 자체는 추가 원본 대조 없이 독립적으로 증명할 수 없다.
- 시간/설비/로트 독립성은 미확인이다. 이미 전체 데이터 EDA를 수행했으므로 Test의 엄격한 미노출성은 제한된다.
- 기존 Logistic Regression 검증·Test 평가는 별도 파이프라인으로 완료했다. 이 문서 뒤의 새 비지도 비교는
  해당 산출물이나 고정된 Test 평가를 변경하지 않는다.

## 검증 상태

- 소규모 합성 입력 기반 신규 단위 테스트 3개 통과 (로컬 임시 환경, scikit-learn 1.8).
- 기존 기준 모델 학습 및 Validation/Test 평가는 artifacts/modeling, artifacts/evaluation,
  artifacts/final_holdout에 기록되어 있다.

## 비지도 모델 비교

### 실행과 비교 조건

`compare_unsupervised.py`는 기존 `artifacts/preprocessing`의 11개 특징을 사용한다. Train 특징만 모델별
`fit`에 전달하며 Train 라벨은 파싱하지 않는다. 네 모델의 Validation 점수 산출을 마친 뒤에만 Validation
Machine failure 라벨을 읽어 평가한다. 입력으로 여는 CSV는 Train/Validation 특징과 추적정보 4개이며,
Test CSV와 Test 행이 있는 split manifest는 열지 않는다.

|모델|설정|이상 점수|
|---|---|---|
|Isolation Forest|`n_estimators=200`, `contamination=auto`, `random_state=42`|`-decision_function`|
|LOF|`n_neighbors=20`, `novelty=True`, `contamination=auto`|신규 Validation 행의 `-score_samples`|
|One-Class SVM|`nu=0.05`, `kernel=rbf`, `gamma=scale`|`-decision_function`|
|PCA|`n_components=0.95`, `svd_solver=full`|재구성 평균제곱오차|

점수는 모두 클수록 이상에 가깝다. AP를 주 선정 지표로 사용하고 AP 동률일 때 ROC-AUC, Recall, 낮은 FPR,
짧은 총 모델 실행시간을 순서대로 적용한다. 모델별 임계값은 Validation 점수의 95분위수로 고정했다.
이는 라벨을 사용하지 않는 임시 5% 후보 검토 가정이며, 동률 점수 때문에 실제 표시 비율이 5%를 넘을 수 있다.

### AI4I 실행 결과

Train 6,000행, Validation 2,000행(고장 라벨 68건)을 사용한 결과다. 임계값 성능은 각 모델의 95분위수에서
측정했다.

|순위|모델|AP|ROC-AUC|Precision|Recall|FPR|학습 초|Validation 점수 초|
|---:|---|---:|---:|---:|---:|---:|---:|---:|
|1|One-Class SVM|0.277678|0.852203|0.240000|0.352941|0.039337|0.146608|0.067527|
|2|LOF|0.197021|0.842498|0.190000|0.279412|0.041925|0.267622|0.086561|
|3|PCA|0.189558|0.747755|0.170000|0.250000|0.042961|0.004389|0.001278|
|4|Isolation Forest|0.154999|0.807659|0.190000|0.279412|0.041925|0.364237|0.055622|

One-Class SVM을 이번 네 비지도 모델 중 후속 검토 후보로 선정했다. AP가 가장 높고 이 비교의 95분위수
기준에서 Recall 0.352941, FPR 0.039337이었다. 이는 기존 Logistic Regression과의 재선정이나 운영 승인,
고장 원인 해석이 아니다. Machine failure 라벨은 일부 알려진 고장만 나타내며, 이상 전체의 정답 라벨은 아니다.

### 산출물 및 검증

- `artifacts/unsupervised_comparison/`에 모델별 `.joblib`, `validation_scores.csv`, `metrics.csv`,
  AP순 `comparison.csv`, 합쳐진 `pr_curves.csv`, `precision_recall_curves.png`, `summary.json`, `report.md`를 저장했다.
- `summary.json`에는 설정, scikit-learn 버전, 입력 4개 CSV의 SHA-256, 모델별 학습·점수 시간, 평가 지표,
  선택 결과 및 Test 미접근 기록이 있다.
- 합성 테스트는 fit 호출에 라벨이 전달되지 않는 점, Validation 라벨을 읽는 시점, LOF novelty 점수 방향,
  네 점수 정의, 재현성, 입력 검증 및 Test 파일 미접근을 검사한다.
