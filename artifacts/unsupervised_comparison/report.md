# 비지도 이상탐지 모델 비교

Train 6,000행의 11개 특징으로 학습하고, Validation 2,000행에서 점수를 산출했다. Validation Machine failure 양성은 68건이다.
학습에는 특징값만 전달했다. 모든 Validation 점수를 산출한 뒤에만 라벨을 읽어 평가했다. Test 입력·라벨·매니페스트는 읽거나 평가하지 않았다.

## 비교 결과

모델은 AP 내림차순으로 정렬했다. 임계값은 모델별 Validation 점수의 95분위수이며, 라벨 기반으로 조정하지 않았다.

|순위|모델|AP|ROC-AUC|Recall|FPR|Precision|F1|F2|표시 비율|학습 초|점수 초|선정|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
|1|one_class_svm|0.277678|0.852203|0.352941|0.039337|0.240000|0.285714|0.322581|5.000%|0.146608|0.067527|예|
|2|lof|0.197021|0.842498|0.279412|0.041925|0.190000|0.226190|0.255376|5.000%|0.267622|0.086561||
|3|pca|0.189558|0.747755|0.250000|0.042961|0.170000|0.202381|0.228495|5.000%|0.004389|0.001278||
|4|isolation_forest|0.154999|0.807659|0.279412|0.041925|0.190000|0.226190|0.255376|5.000%|0.364237|0.055622||

## 최종 후보

**one_class_svm**를 후속 검토 후보로 선정했다. AP 0.277678가 네 모델 중 가장 높았다. AP를 주 지표로 두고 ROC-AUC, Recall, FPR 및 학습·점수 산출시간을 보조 지표로 확인했다. AP가 같으면 ROC-AUC, Recall, 낮은 FPR, 짧은 합산 실행시간 순으로 결정한다. 이는 후보 선정이며 운영 적용 또는 고장 원인 판정을 의미하지 않는다.

## 점수와 임계값

- Isolation Forest: `-decision_function`.
- LOF: `novelty=True`로 학습하고 Validation 신규 관측에 `-score_samples` 적용.
- One-Class SVM: `-decision_function`.
- PCA: 재구성 평균제곱오차(mean squared reconstruction error).
- 모든 점수는 클수록 이상에 가깝다. PR 곡선은 `pr_curves.csv`와 `precision_recall_curves.png`에 저장했다.
- 95분위수 임계값은 라벨 없는 임시 5% 후보 검토 가정이다. 임계값 점수 동률로 실제 표시 비율은 5%보다 높을 수 있다.

## 설정과 산출물

- 모델별 `.joblib`, `validation_scores.csv`, `metrics.csv`, `comparison.csv`, PR 곡선, `summary.json` 및 이 보고서를 저장했다.
- `summary.json`에 입력 파일 SHA-256, 설정, scikit-learn 버전, 학습·점수 시간과 결과 요약을 기록했다. 해시는 Train/Validation 특징 및 추적 CSV만 포함한다.
- 재실행: `python compare_unsupervised.py --input artifacts/preprocessing --output artifacts/unsupervised_comparison_new`

## 한계

Machine failure 라벨은 평가용 대리 라벨일 뿐 이상 원인이나 모든 유형의 공정 이상을 뜻하지 않는다. 라벨 불일치와 무작위 분할의 시간·설비 독립성은 기존 한계로 남는다. Validation으로 모델을 선정했으므로 독립 Test 평가 전의 후보 결과다.
