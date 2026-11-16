# AI4I 라벨 정합성 검증 작업

저장소 루트에 `scripts/`, `tests/` 폴더를 병합합니다. **기존 파일이 있으면 먼저 diff를 검토하세요.**

## 실행

```bash
python scripts/validate_ai4i_labels.py --input ai4i2020.csv
python -m unittest discover -s tests -v
```

CSV 경로가 다르면 `--input`만 조정하세요.

## 생성 파일

- `artifacts/validation/label_audit.json`: 데이터 해시·고장 유형 수·불일치 유형별 건수
- `artifacts/validation/label_discrepancies.csv`: 불일치 행의 UID/UDI 및 플래그

## 후속 점검

1. 실행 후 불일치가 이전 EDA의 27건(18+9)과 일치하는지 확인
2. RNF 발생 건수가 이전 EDA의 19건과 같은지 확인
3. UCI 공식 문서와 차이를 확인하고, 확정 사실과 미확인 사항을 분리
4. 라벨을 임의로 수정·삭제하지 않고 원본 `Machine failure`를 일단 별도로 유지
5. 검증 결과와 라벨 처리 방침을 `PLANS.md`에 기록한 뒤 전처리 단계 시작

테스트는 합성 예제로만 수행합니다. 실제 AI4I CSV의 결과를 확인하지 않은 상태입니다.

## 커밋

```bash
git add scripts/validate_ai4i_labels.py tests/test_validate_ai4i_labels.py artifacts/validation PLANS.md
git commit -m "test: audit AI4I failure label consistency"
git push origin main
```
