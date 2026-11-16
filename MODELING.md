# AI4I 모델링: 기준 모델과 이상 점수

## 실행

`python train_models.py --input artifacts/preprocessing --output artifacts/modeling`

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
- 다음 단계: Validation에서 AP/PR-AUC, ROC-AUC, Precision/Recall/F1 및 임계값을 결정하고,
  선택한 모델과 임계값을 고정한 후 Test에서 최종 평가한다.

## 검증 상태

- 소규모 합성 입력 기반 신규 단위 테스트 3개 통과 (로컬 임시 환경, scikit-learn 1.8).
- 저장소의 AI4I 10,000건으로 실제 학습 실행 및 기존 전체 테스트 재실행은 아직 확인되지 않았다.
