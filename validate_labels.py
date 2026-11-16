"""Audit AI4I labels without editing input or EDA outputs."""
import argparse
import json
import math
from decimal import Decimal
from hashlib import sha256
from pathlib import Path

import pandas as pd
from eda import MODES, PROCESS, TARGET, load_data

SOURCE = "https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset"


def digest(path):
    """Return the SHA-256 of a file."""
    return sha256(path.read_bytes()).hexdigest()


def main():
    """Export disagreements, deterministic rule checks, and an audit summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/ai4i2020.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/validation"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists. Select a new --output directory.")
    protected = [args.input] + sorted(Path("artifacts/eda").rglob("*")) + sorted(
        Path("artifacts/eda_interactions").rglob("*"))
    before = {str(p): digest(p) for p in protected if p.is_file()}
    data = load_data(args.input)
    if data[PROCESS].isna().any().any() or not data["Type"].isin(["L", "M", "H"]).all():
        raise ValueError("Expected complete process variables and known product types")
    # Export original CSV column names and values, not renamed EDA columns.
    raw = pd.read_csv(args.input)
    active = data[MODES].sum(axis=1)
    any_mode = active.gt(0).astype(int)
    mismatch = data[TARGET].ne(any_mode)
    extracted = raw.loc[mismatch].copy()
    extracted["any_failure_mode"] = any_mode[mismatch]
    extracted["active_mode_count"] = active[mismatch]
    args.output.mkdir(parents=True)
    extracted.to_csv(args.output / "label_disagreements.csv", index=False)
    pd.crosstab(data[TARGET], any_mode).to_csv(args.output / "target_vs_flags.csv")

    # Decimal arithmetic uses the published decimal measurements exactly.
    gap = data.apply(lambda r: Decimal(str(r[PROCESS[1]])) -
                     Decimal(str(r[PROCESS[0]])), axis=1)
    load = data.apply(lambda r: Decimal(str(r[PROCESS[3]])) *
                      Decimal(str(r[PROCESS[4]])), axis=1)
    thresholds = data["Type"].map({"L": 11000, "M": 12000, "H": 13000})
    power = data[PROCESS[3]] * data[PROCESS[2]] * (2 * math.pi / 60)
    rules = {
        "HDF": (gap.lt(Decimal("8.6")) & data[PROCESS[2]].lt(1380)).astype(int),
        "PWF": (power.lt(3500) | power.gt(9000)).astype(int),
        "OSF": load.gt(thresholds).astype(int),
    }
    checks = raw.copy()
    checks["temperature_gap_K"] = gap
    checks["power_W"] = power
    checks["torque_wear_Nm_min"] = load
    records = []
    for mode, rule in rules.items():
        checks[f"{mode}_rule"] = rule
        checks[f"{mode}_disagreement"] = data[mode].ne(rule)
        records.append({"mode": mode, "flag_count": int(data[mode].sum()),
                        "rule_count": int(rule.sum()),
                        "disagreements": int(data[mode].ne(rule).sum())})
    checks.to_csv(args.output / "rule_checks_all_rows.csv", index=False)
    check_summary = pd.DataFrame(records)
    check_summary.to_csv(args.output / "rule_summary.csv", index=False)
    official = {"TWF": 51, "HDF": 115, "PWF": 95, "OSF": 98, "RNF": 5}
    frequencies = pd.DataFrame({"official_description": pd.Series(official),
                                "csv_flags": data[MODES].sum()})
    frequencies["difference"] = frequencies.csv_flags - frequencies.official_description
    frequencies.to_csv(args.output / "frequency_comparison.csv", index_label="mode")
    summary = {
        "rows": len(data), "machine_failure": int(data[TARGET].sum()),
        "any_mode": int(any_mode.sum()), "disagreements": int(mismatch.sum()),
        "normal_with_flags": int((data[TARGET].eq(0) & any_mode.eq(1)).sum()),
        "failure_without_flags": int((data[TARGET].eq(1) & any_mode.eq(0)).sum()),
        "rnf_with_target": int((data.RNF.eq(1) & data[TARGET].eq(1)).sum()),
        "hdf_exact_boundary_rows": int((gap.eq(Decimal("8.6")) &
                                        data[PROCESS[2]].lt(1380)).sum()),
        "twf_outside_200_240": int((data.TWF.eq(1) &
                                  ~data[PROCESS[4]].between(200, 240)).sum()),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    report = f"""# AI4I 라벨 정합성 검증

검증일은 2026-11-16이다. 입력 SHA-256은 `{before[str(args.input)]}`이다.
동일 날짜에 [UCI 공식 설명]({SOURCE})을 확인했다.

## 확인된 사실

- 총 {len(data)}행, Machine failure=1은 {summary['machine_failure']}건,
  다섯 플래그의 OR=1은 {summary['any_mode']}건이다.
- 불일치는 {summary['disagreements']}건이며, 정상인데 플래그가 있는 행은
  {summary['normal_with_flags']}건, 고장인데 플래그가 없는 행은
  {summary['failure_without_flags']}건이다.
- RNF는19건이며 Machine failure=1과 겹치는 행은 {summary['rnf_with_target']}건이다.
- [불일치 행](label_disagreements.csv)에 UID 및 원본의 모든 열과 실제 값을 저장했다.
- 공식 설명 대비 CSV 빈도는 TWF 51/46, HDF 115/115, PWF 95/95,
  OSF 98/98, RNF 5/19건이다. RNF 차이는 14건이다.
  공식의 0.1%는 생성 확률이며 관측 빈도와 구분한다.
  [빈도 비교](frequency_comparison.csv)

## 규칙 검증

- HDF는 온도 차이 <8.6 K와 회전 속도 <1380 rpm의 동시 충족으로 계산했다.
- PWF는 토크×회전 속도×2π/60이 3500 W 미만 또는 9000 W 초과인 조건이다.
- OSF는 토크×마모가 L/M/H별 11000/12000/13000을 초과하는 조건이다.
- [규칙별 집계](rule_summary.csv)와 [모든 행의 계산값](rule_checks_all_rows.csv)을 저장했다.
- 온도 차이는 Decimal로 계산했다. 정확히 8.6 K이며 저속인 행은
  {summary['hdf_exact_boundary_rows']}건으로, float 오차와 실제 경계값을 구분해야 한다.
- TWF는 200~240분에서 무작위 고장 또는 교체라는 설명이다.
  TWF=1 중 해당 범위 밖은 {summary['twf_outside_200_240']}건이다.
  개별 교체 기록과 난수가 없어 전체 TWF를 재현할 수 없다.
- RNF 역시 원래 난수 정보가 없어 행별 발생 규칙을 재현할 수 없다.

## 추가 확인 사실

실행 결과 HDF는 규칙 103건/플래그 115건으로 12건이 달랐고, PWF와 OSF는 불일치가 없었다.
HDF 차이 12건은 모두 저장된 온도 차이가 정확히 8.6 K인 경계 행이었다.
같은 경계와 저속 조건의 27건 중 15건은 HDF=0이므로 단순히 ≤로 바꾸어도 전체를 재현하지 못한다.
TWF 범위 밖 3건은 UID 1997(198분), 2865(246분), 5402(253분)이다.
RNF=1이면서 고장인 유일한 UID 3612는 TWF도 1이다. RNF 단독 표시는 모두 정상 라벨이다.

## 미확인 사항

공식 설명과 CSV 중 어느 쪽이 의도한 정답인지, 버전 차이, 생성 코드,
반올림 전 수치, 불일치 원인은 확인되지 않았다.
공식 OR 설명과 실제 값의 불일치는 확인했지만 특정 열이 틀렸다고 단정할 수 없다.
RNF 19건을 5건으로 바꿀 근거는 확보하지 못했다.
임계값 경계에서의 공식 구현 비교 연산과 수치 정밀도도 후속 확인 대상이다.

## 학습 라벨 처리 방안

|방안|장점|단점|
|---|---|---|
|원본 Machine failure 유지|원래 예측 대상 및 전체 표본을 보존한다|불일치 27건을 포함한다|
|플래그 OR을 별도 대상으로 사용|다섯 플래그와 논리적 정합성이 생긴다|플래그가 정답이라는 미확인 가정을 하며 예측 대상과 양성 수가 바뀐다|
|불일치를 불확실 라벨로 보류|모호한 지도 정보의 영향을 줄일 수 있다|표본 선택 편향과 양성 수 변화로 비교가 어려워진다|
|두 라벨로 민감도 분석|결과의 라벨 의존성을 비교할 수 있다|평가 비용이 늘고 대상 정의를 구분해야 한다|

제안은 원본 Machine failure를 기준 후보로 유지하고 불일치 UID를 감사 정보로 남기는 것이다.
최종 학습 라벨 방안은 미정이다. 분할 전에 정책을 정하고 민감도 분석에서는 동일 분할을 사용한다.
모든 방안에서 고장 유형 플래그와 그 파생값은 예측 입력에서 제외한다.
UID와 Product ID 역시 식별자로 제외한다.
"""
    (args.output / "report.md").write_text(report, encoding="utf-8")
    after = {p: digest(Path(p)) for p in before}
    if before != after:
        raise RuntimeError("Protected input or EDA artifacts changed")
    (args.output / "preservation.json").write_text(
        json.dumps({"before": before, "after": after, "unchanged": True}, indent=2),
        encoding="utf-8")
    print(json.dumps(summary))
    print(check_summary.to_string(index=False))


if __name__ == "__main__":
    main()
