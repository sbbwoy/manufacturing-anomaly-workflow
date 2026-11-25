# 2026-11-25 비지도 이상탐지 모델 비교 개발 보고서

## 목표와 범위

기존 전처리 산출물의 Train/Validation 11개 특징을 사용해 Isolation Forest, LOF, One-Class SVM, PCA를
같은 조건으로 비교했다. 기존 Logistic Regression, Validation/Test 평가 코드와 산출물은 수정하지 않았다.
새 실행 결과는 `artifacts/unsupervised_comparison/`에 분리했다.

## 구현과 데이터 접근

- `compare_unsupervised.py`는 Train/Validation 특징과 추적 CSV 네 개만 입력으로 연다. split manifest와
  Test 파일은 사용하지 않았다.
- Train 추적정보를 읽을 때 `Machine failure` 열을 제외하고, 모델마다 Train 특징 행렬만 `fit`에 전달한다.
- 네 모델의 Validation 점수 산출을 모두 마친 뒤 Validation 라벨을 읽어 평가했다.
- Isolation Forest는 `-decision_function`, novelty LOF는 신규 Validation 행의 `-score_samples`,
  One-Class SVM은 `-decision_function`, PCA는 재구성 평균제곱오차를 사용한다. 점수가 클수록 이상에 가깝다.
- 각 모델의 임계값은 Validation 점수 95분위수로 계산했다. 임계값을 라벨에 맞춰 조정하지 않았다.
  이는 라벨 없는 임시 5% 후보 검토 가정이며 점수 동률 시 표시 비율은 5%보다 커질 수 있다.
- 모델별 학습시간과 Validation 점수 산출시간, 입력 CSV SHA-256, sklearn 버전과 설정을 기록했다.

## 비교 결과

Train 6,000행과 Validation 2,000행(원본 Machine failure 양성 68건)을 평가했다.

|모델|AP|ROC-AUC|Precision|Recall|F1|F2|FPR|학습 초|Validation 점수 초|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|Isolation Forest|0.154999|0.807659|0.190000|0.279412|0.226190|0.255376|0.041925|0.364237|0.055622|
|LOF|0.197021|0.842498|0.190000|0.279412|0.226190|0.255376|0.041925|0.267622|0.086561|
|One-Class SVM|**0.277678**|**0.852203**|**0.240000**|**0.352941**|**0.285714**|**0.322581**|**0.039337**|0.146608|0.067527|
|PCA|0.189558|0.747755|0.170000|0.250000|0.202381|0.228495|0.042961|**0.004389**|**0.001278**|

이번 네 비지도 모델 중 One-Class SVM을 후속 검토 후보로 선정했다. AP가 가장 높았고, 95분위수 기준 Recall도
가장 높으며 FPR은 가장 낮았다. 총 학습·점수 시간은 약 0.214초다. PCA가 가장 빨랐지만 AP 0.189558,
ROC-AUC 0.747755로 순위 지표는 One-Class SVM보다 낮았다.

이 선택은 비지도 네 모델 사이의 선정이다. 기존 Validation 평가에서 Logistic Regression의 AP는 0.421178로
더 높았으므로 이번 결과는 기존 지도 모델 선택이나 이미 고정된 Test 평가를 대체하지 않는다.

## 산출물

`artifacts/unsupervised_comparison/`에 모델별 `.joblib`, `validation_scores.csv`, 모델 지표 `metrics.csv`,
AP 순위 비교표 `comparison.csv`, `pr_curves.csv`, `precision_recall_curves.png`, 실행 설정·해시·시간·요약의
`summary.json`, 그리고 상세 `report.md`를 저장했다.

## 검증

- 신규 합성 테스트 3개 통과: 모델 fit 라벨 미전달, Validation 라벨 후행 로드, LOF `novelty=True`와 점수 방향,
  네 점수 정의, 재현성, 입력 스키마 경계 사례, 95분위수 임계값의 라벨 독립성 및 Test 파일 접근 차단.
- 전체 테스트: 25개 통과. 기존 테스트는 합성 fixture를 사용했고 운영 Test CSV를 읽지 않았다.
- 기존 경로 `artifacts/modeling/`, `artifacts/evaluation/`, `artifacts/final_holdout/`과 기존 모델·평가 스크립트는
  Git 상태에서 변경되지 않았다.

## 한계

Machine failure는 평가용 대리 라벨이며 모든 공정 이상을 대표하지 않는다. 비지도 점수는 고장 확률이 아니고
원인을 설명하지 않는다. 95분위수 기준은 현장 검토 비용과 업무량으로 검증되지 않은 임시 가정이다. 시간·설비·로트
독립성 및 라벨 불일치 원인은 기존 한계로 남는다. 이번 비교는 Test 성능을 평가하지 않았다.
