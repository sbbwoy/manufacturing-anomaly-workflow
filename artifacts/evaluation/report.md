# AI4I Validation 성능 평가

Validation 2000행, 고장 68건, 라벨 불일치 6건을 평가했다.
입력은 기존 validation_scores.csv 한 파일이며 재학습하지 않았다.
재실행 명령은 python evaluate_validation.py --output artifacts/evaluation_new 이다.

## 모델 비교와 임계값

AP는 average_precision_score의 비보간 평균 정밀도이며, 사다리꼴 PR 면적과 구분한다.
임계값은 관측된 고유 점수 중 F2 최대값을 선택했다. 동률이면 높은 임계값을 선택한다.
점수 ≥ 임계값이면 고장으로 분류한다. F2는 Recall에 더 큰 가중치를 둔다.

- logistic_regression — AP 0.421178, ROC-AUC 0.918821, 임계값 0.721824582813, Precision 0.284884, Recall 0.720588, F1 0.408333, F2 0.551802. TN/FP/FN/TP = 1809/123/19/49.
- isolation_forest — AP 0.154999, ROC-AUC 0.807659, 임계값 0.0545467131195, Precision 0.153409, Recall 0.397059, F1 0.221311, F2 0.301339. TN/FP/FN/TP = 1783/149/41/27.

## 라벨 민감도

임계값을 고정하고 원본 라벨, 불일치 관측 제외, 플래그 OR 대체 라벨을 비교했다.
OR 대체 라벨은 기존 감사 불일치 여부를 이용해 해당 이진 라벨만 반전했다.
원본 라벨과 행은 수정하지 않았다. 대체 라벨의 정답 여부는 미확인이다.

- logistic_regression / original — 2000행, 양성 68건, Precision 0.284884, Recall 0.720588, F2 0.551802, FP/FN 123/19.
- logistic_regression / exclude_disagreements — 1994행, 양성 65건, Precision 0.286550, Recall 0.753846, F2 0.568445, FP/FN 122/16.
- logistic_regression / flag_or_alternative — 2000행, 양성 68건, Precision 0.290698, Recall 0.735294, F2 0.563063, FP/FN 122/18.
- isolation_forest / original — 2000행, 양성 68건, Precision 0.153409, Recall 0.397059, F2 0.301339, FP/FN 149/41.
- isolation_forest / exclude_disagreements — 1994행, 양성 65건, Precision 0.153409, Recall 0.415385, F2 0.309633, FP/FN 149/38.
- isolation_forest / flag_or_alternative — 2000행, 양성 68건, Precision 0.153409, Recall 0.397059, F2 0.301339, FP/FN 149/41.

## 저장 결과와 한계

- metrics.csv에 AP, ROC-AUC, 임계값 지표와 혼동행렬을 저장했다.
- 모델별 errors.csv에 UID, 실제 라벨, 점수, FP/FN 및 불일치 여부를 저장했다.
- sensitivity.csv에 각 시나리오의 전체 지표를 저장했다.
- 모델별 pr_curve.csv에는 임계값별 Precision/Recall을 저장했다.
- 같은 Validation에서 임계값을 선택하고 평가했으므로 성능에 선택 편향이 있다.
- 기존 전체 데이터 EDA와 무작위 분할의 시간·설비 독립성 한계가 남는다.
- 고장 라벨은 원본 Machine failure이며 원인이나 실제 운영 성능을 단정할 수 없다.
- Test 데이터는 이번 실행에서 읽거나 평가하지 않았다. 최종 평가가 후속 과제다.
