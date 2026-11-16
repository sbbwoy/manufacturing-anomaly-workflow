# AI4I 전처리 정책

## 실행

Python 환경에 pandas, numpy, scikit-learn, joblib, cloudpickle 및 기존 EDA의 matplotlib,
seaborn이 필요하다. 테스트는 표준 라이브러리 unittest를 사용한다.

`python preprocess.py --input data/ai4i2020.csv --output artifacts/preprocessing`

`python -m unittest discover -s tests -p "test_*.py"`

이미 존재하는 출력 폴더는 거부한다. 재실행 시 새 출력 경로를 지정한다.
성공한 결과만 임시 폴더에서 최종 경로로 이동한다.

## 라벨과 감사 정보

원본 Machine failure를 기준 라벨로 유지한다. 불일치 행을 포함한 모든 행을 보존한다.
기존 scripts/validate_ai4i_labels.py의 audit 함수를 재사용해 감사 파일을
새 실행 폴더의 audit/에 저장한다. 두 검증 스크립트의 고장 규칙을 재구현하지 않는다.
기존 검증 보고서가 설명하는 불일치 원인은 여전히 미확인이다.

## 스키마와 파생변수

- CSV 입력은 UID, Product ID, Type, 다섯 수치 공정 변수, Machine failure와 다섯 고장 플래그가 필요하다.
  기존 eda.load_data를 재사용해 단위가 없는 원본 공정 열 이름을 단위 포함 이름으로 정규화한다.
- Pipeline의 transform 입력에는 Type과 다섯 공정 변수만 필요하다. 추가 열은 버린다.
- Type은 L/M/H만 허용하며 누락과 미지의 값은 오류다.
- 수치 열은 숫자 자료형이어야 한다. 결측값, 무한대, 불리언은 거부한다.
  임의 대체나 이상치 삭제를 수행하지 않는다. UID는 고유해야 하며 두 식별자는 결측을 허용하지 않는다.
- 온도 차이는 공정 온도 − 공기 온도(K)다.
- 기계적 동력은 토크(Nm) × 회전 속도(rpm) × 2π/60(W)다.
- 토크×공구 마모의 단위는 Nm·min이다.

파생변수는 행별 수식으로 계산한다. 라벨 검증의 경계 비교와 달리
이번 목적은 연속값 전처리이며 부동소수점 연산을 사용한다.
고장 유형 플래그와 그 파생값, Machine failure 및 식별자는 입력에 포함하지 않는다.

## 분할과 학습 범위

원본 행 위치를 stratify=Machine failure, random_state=42로 먼저 60/40으로 분할하고,
보류한 40%를 동일 조건으로 절반씩 나눠 Validation/Test로 지정한다.
정수 표본 할당 때문에 각 집합의 고장률에는 작은 차이가 생길 수 있다.
분할 전에 전처리 통계량을 학습하지 않는다.

Pipeline은 행별 특징 생성과 ColumnTransformer로 구성한다.
Type은 고정 범주 L/M/H의 OneHotEncoder로 3열을 만들고,
수치 원변수 5개와 파생변수 3개는 StandardScaler로 변환한다.
출력은 총 11열이다. fit은 Train에만 호출하고 Validation/Test에는 transform만 호출한다.
추론에도 저장한 Pipeline을 재사용하며 joblib 파일은 신뢰하는 자체 산출물만 읽는다.

## 저장 형식과 추적

각 split의 features.csv와 tracking.csv는 행 순서로 대응한다.
tracking에는 UID, Product ID, 원본 라벨, source_row(0부터), csv_row(헤더 포함 2부터),
불일치 여부와 split을 저장한다. split_manifest.csv는 원래 행 순서로 전체 배정을 기록한다.
summary.json은 열 이름, 시드, 버전, 집합별 건수와 고장 수를 담는다.
preprocessor.joblib은 Train에서 학습한 전처리 객체다.
preservation.json은 원본 및 실행 이전 artifacts 파일의 해시 보존 검증을 기록한다.

## 한계

요청된 무작위 계층 분할을 적용했다. 실제 시간·설비·로트 독립성은 확인되지 않았으며,
현재 분할로 미래 설비 운전 환경에 대한 일반화를 판단할 수 없다.
전체 데이터 EDA가 이미 수행되었으므로 Test는 탐색에 전혀 노출되지 않은 집합이 아니다.
새로운 Type·결측 입력의 처리 정책 확대와 실제 예측 시점의 변수 가용성은 후속 과제다.
