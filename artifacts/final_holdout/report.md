# AI4I 최종 Test 평가

Validation AP 최고 모델 Logistic Regression과 Validation F2 최대화 임계값
0.7218245828133553을 Test 접근 전에 config/final_holdout.json에 고정했다.
설정 SHA-256은 `8c1db776be61a40010c248f79a1786e95d97e386f14ef9c231e292d45cd55a9e`이다.

## 실제 실행 결과

- Test 2000건, 원본 Machine failure 양성 68건
- AP(비보간 평균 정밀도) 0.412090, ROC-AUC 0.925428
- Precision 0.328244, Recall 0.632353
- F1 0.432161, F2 0.533499
- 혼동행렬 TN/FP/FN/TP는 1844/88/25/43
- 라벨 불일치 관측 6건을 보존했다.
- 저장된 모델의 predict_proba를 1회 호출했다. 재학습과 임계값 조정을 수행하지 않았다.
- UID별 예측은 test_predictions.csv, 오탐·미탐은 test_errors.csv에 저장했다.

## 검증과 제한

입력 스키마와 UID·원본 행 위치를 split_manifest.csv와 대조했다.
전체 manifest에서 UID·행 위치의 중복을 검사했다. 기존 load_split과 평가 metrics 함수를 재사용했다.
추적정보 정합성과 별개로 수치 특징과 UID의 대응은 기존 전처리 생성 순서에 의존한다.
원본 CSV, 고정 설정 및 기존 산출물의 SHA-256을 실행 전후 비교해 보존을 확인했다.
기존 학습 모델은 고정 해시로 검증했다. Test 입력 해시는 이번 첫 접근에서 기록했다.

중복 실행은 artifacts/final_holdout_execution.json의 배타적 생성으로 차단한다.
출력 경로 변경으로 재평가할 수 없으며, 실패 시에도 실행 기록을 남긴다.
기록을 수동 삭제하거나 다른 저장소 사본에서 실행하는 행위까지 차단하는 전역 서비스는 아니다.

원본 라벨 불일치의 원인은 미확인이다. 전체 데이터 EDA가 이미 Test를 포함했다.
시간·설비 독립성은 검증되지 않았다. 따라서 이 결과를 완전히 미노출된 운영 성능이나
실제 고장의 인과관계로 해석할 수 없다. 이번 Test 결과에 기반한 재선택·재조정은 수행하지 않는다.
