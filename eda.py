"""Run AI4I exploratory analysis and save aggregate tables and figures."""
import argparse
from hashlib import sha256
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

TARGET = "Machine failure"
MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
IDS = ["UID", "Product ID"]
PROCESS = ["Air temperature [K]", "Process temperature [K]",
           "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]


def load_data(path):
    """Read the original CSV and reject invalid analysis columns or labels."""
    data = pd.read_csv(path)
    for column in PROCESS:
        bare_name = column.split(" [", 1)[0]
        if bare_name in data and column not in data:
            data = data.rename(columns={bare_name: column})
    missing = set(IDS + ["Type"] + PROCESS + [TARGET] + MODES) - set(data)
    if missing or data.empty:
        raise ValueError(f"Expected nonempty AI4I input. Missing columns: {sorted(missing)}")
    for column in PROCESS:
        if not pd.api.types.is_numeric_dtype(data[column]):
            raise ValueError(f"Expected numeric values in {column}")
    for column in [TARGET] + MODES:
        if not data[column].isin([0, 1]).all():
            raise ValueError(f"Expected nonmissing binary labels in {column}")
    return data


def save_figure(fig, output, name):
    """Save and close a figure."""
    fig.tight_layout()
    fig.savefig(output / name, dpi=150)
    plt.close(fig)


def analyze(data, output):
    """Save aggregate statistics and render figures for the six EDA steps."""
    schema = pd.DataFrame({"dtype": data.dtypes.astype(str),
                           "missing": data.isna().sum(), "unique": data.nunique()})
    schema.to_csv(output / "schema.csv", index_label="column")
    data.drop(columns=IDS).describe(include="all").T.to_csv(output / "descriptive.csv")
    counts = data[TARGET].value_counts().reindex([0, 1], fill_value=0)
    pd.DataFrame({"count": counts, "percent": 100 * counts / len(data)}).to_csv(
        output / "target.csv")
    fig, ax = plt.subplots()
    bars = ax.bar(["Normal", "Failure"], 100 * counts / len(data))
    ax.bar_label(bars, labels=[f"{n:,} ({n / len(data):.2%})" for n in counts])
    ax.set(ylabel="Percent of records", ylim=(0, 110))
    save_figure(fig, output, "target.png")

    group_stats = data.groupby(TARGET)[PROCESS].agg(["count", "mean", "median", "std"])
    group_stats.to_csv(output / "process_by_target.csv")
    type_counts = pd.crosstab(data["Type"].fillna("(missing)"), data[TARGET]).reindex(
        columns=[0, 1], fill_value=0)
    type_counts["failure_percent"] = 100 * type_counts[1] / (type_counts[0] + type_counts[1])
    type_counts.to_csv(output / "type_by_target.csv")
    fig, axes = plt.subplots(5, 2, figsize=(12, 16))
    labels = data[TARGET].map({0: "Normal", 1: "Failure"})
    for column, (overall, grouped) in zip(PROCESS, axes):
        sns.histplot(data[column], bins=30, ax=overall)
        sns.boxplot(x=data[column], y=labels, order=["Normal", "Failure"], ax=grouped)
        grouped.set_ylabel("")
    save_figure(fig, output, "process_distributions.png")
    fig, ax = plt.subplots()
    sns.barplot(x=type_counts.index, y=type_counts["failure_percent"], ax=ax)
    ax.set(ylabel="Failure within type (%)", xlabel="Product type")
    save_figure(fig, output, "type_failure_rate.png")

    correlations = data[PROCESS].corr()
    correlations.to_csv(output / "pearson.csv")
    data[PROCESS].corr(method="spearman").to_csv(output / "spearman.csv")
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(correlations, annot=True, fmt=".2f", vmin=-1, vmax=1,
                center=0, cmap="vlag", ax=ax)
    save_figure(fig, output, "pearson.png")

    flags = data[MODES].astype("int64")
    mode_count = flags.sum(axis=1)
    mode_frequency = flags.sum()
    pd.DataFrame({"count": mode_frequency, "percent": 100 * mode_frequency / len(data),
                  "target_0": flags.loc[data[TARGET].eq(0)].sum(),
                  "target_1": flags.loc[data[TARGET].eq(1)].sum()}).to_csv(
        output / "failure_modes.csv", index_label="mode")
    cooccurrence = flags.T @ flags
    cooccurrence.to_csv(output / "mode_pairs.csv")
    flags.value_counts().rename("count").to_csv(output / "mode_combinations.csv")
    cross = pd.crosstab(mode_count, data[TARGET]).reindex(
        index=range(len(MODES) + 1), columns=[0, 1], fill_value=0)
    cross.index.name = "active_modes"
    cross.to_csv(output / "mode_count_by_target.csv")
    fig, (freq_ax, pair_ax) = plt.subplots(1, 2, figsize=(11, 4))
    sns.barplot(x=mode_frequency.index, y=mode_frequency, ax=freq_ax)
    freq_ax.set_ylabel("Records with flag")
    sns.heatmap(cooccurrence, annot=True, fmt="d", cmap="Blues", ax=pair_ax)
    pair_ax.set_title("Co-occurrence (diagonal = frequency)")
    save_figure(fig, output, "failure_modes.png")
    return counts, group_stats, correlations, mode_frequency, cross


def write_report(data, source, output, results):
    """Write a Korean report grounded in this run's aggregates."""
    counts, groups, corr, frequencies, cross = results
    process_lines = []
    if {0, 1}.issubset(groups.index):
        for column in PROCESS:
            normal, failure = groups[(column, "median")].reindex([0, 1])
            process_lines.append(f"- {column} 중앙값은 정상 {normal:g}, 고장 {failure:g}였다.")
    pairs = [(abs(corr.at[left, right]), left, right)
             for index, left in enumerate(PROCESS) for right in PROCESS[index + 1:]
             if pd.notna(corr.at[left, right])]
    relation = "계산 가능한 Pearson 상관계수가 없었다."
    if pairs:
        _, left, right = max(pairs)
        relation = f"절댓값이 가장 큰 Pearson 상관은 {left}와 {right}의 {corr.at[left, right]:.3f}였다."
    frequency_text = ", ".join(f"{mode} {n}건" for mode, n in frequencies.items())
    duplicates_without_ids = data.drop(columns=IDS).duplicated().sum()
    report = f"""# AI4I 2020 EDA 실행 결과

입력 파일은 `{source.name}`이다. SHA-256은 `{sha256(source.read_bytes()).hexdigest()}`이다.
출처는 [UCI AI4I 2020](https://doi.org/10.24432/C5HS5C)이며 라이선스는 CC BY 4.0이다.
한 행을 개별 공정 관측으로 집계했다.
공정 변수의 단위 표기는 분석용 열 이름에만 적용했다.

## 1. 데이터 구조와 품질

- {len(data):,}행, {len(data.columns)}열, 결측 셀 {data.isna().sum().sum():,}개
- 전체 열이 같은 중복 행 {data.duplicated().sum():,}건, 두 식별자를 제외한 중복 행 {duplicates_without_ids:,}건
- UID 중복 {data['UID'].duplicated().sum():,}건, Product ID 중복 {data['Product ID'].duplicated().sum():,}건
- 자료형과 고유값 수는 [schema.csv](schema.csv), 기본 통계는 [descriptive.csv](descriptive.csv)에 저장했다.

## 2. 정상과 고장 비율

정상 {counts[0]:,}건({counts[0] / len(data):.2%}), 고장 {counts[1]:,}건({counts[1] / len(data):.2%}).
[비율 그래프](target.png)

## 3. 공정 변수와 그룹 비교

{chr(10).join(process_lines)}

[공정 변수 분포와 그룹별 박스플롯](process_distributions.png) · [제품 유형별 고장률](type_failure_rate.png)

## 4. 상관관계

{relation} Pearson과 Spearman은 다섯 수치 공정 변수에 대해 계산했다.
[상관 히트맵](pearson.png) · [Spearman 표](spearman.csv)

## 5. 고장 유형과 중복 발생

- 전체 관측 기준 유형별 빈도는 {frequency_text}이다.
- 두 종류 이상이 함께 표시된 관측은 {cross.loc[2:].to_numpy().sum():,}건이다.
- 고장 유형 표시가 있는데 정상 라벨인 관측은 {cross.loc[1:, 0].sum():,}건이다. 유형 표시가 없는데 고장 라벨인 관측은 {cross.loc[0, 1]:,}건이다.
- [유형별 빈도와 쌍별 중복](failure_modes.png), [정확한 조합별 빈도](mode_combinations.csv), [활성 유형 수와 대상 라벨 교차표](mode_count_by_target.csv)를 저장했다.

## 6. 해석과 후속 검증

- 그룹별 중앙값은 관측된 차이이며, 박스플롯에서 분포의 겹침과 산포를 함께 확인할 수 있다. 그룹 차이나 상관계수만으로 고장 원인을 판단할 수 없다.
- 합성 데이터이므로 실제 생산 설비에 적용하기 전에 별도 검증이 필요하다. 분할 전 관측 순서의 의미와 설비·로트 구분 가능성을 확인하고, 고장 희소성을 고려해 평가 지표를 정한다.
- 후속 분석에서는 온도 차이와 회전 속도, 토크와 공구 마모의 조합을 제품 유형별로 확인한다. 예측 시점과 대상 라벨의 관측 시점도 정의한다.
- 유형 표시와 대상 라벨이 다른 행은 원본 상태로 집계했다. 후속 작업에서 라벨 생성 규칙과 각 불일치 사례를 확인한다.
- **데이터 누수 방지** — TWF, HDF, PWF, OSF, RNF 및 그 파생값은 고장 결과를 드러낼 수 있으므로 EDA 전용이다. 향후 Machine failure 예측 입력에서 모두 제외한다. Machine failure는 대상 라벨로만 사용한다.
- UID와 Product ID는 식별자이므로 중복 점검에만 사용하고 예측 입력에서 제외한다. Type과 수치 공정 변수는 예측 시점에 실제로 확보 가능한지 확인한다.
"""
    (output / "report.md").write_text(report, encoding="utf-8")


def main():
    """Run the EDA from the project root or paths supplied by the caller."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/ai4i2020.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/eda"))
    args = parser.parse_args()
    data = load_data(args.input)
    if any((args.output / name).resolve() == args.input.resolve()
           for name in ("schema.csv", "descriptive.csv", "target.csv", "process_by_target.csv",
                        "type_by_target.csv", "pearson.csv", "spearman.csv", "failure_modes.csv",
                        "mode_pairs.csv", "mode_combinations.csv", "mode_count_by_target.csv",
                        "target.png", "process_distributions.png", "type_failure_rate.png",
                        "pearson.png", "failure_modes.png", "report.md")):
        parser.error("Input CSV cannot also be an output file")
    args.output.mkdir(parents=True, exist_ok=True)
    sns.set_theme()
    write_report(data, args.input, args.output, analyze(data, args.output))


if __name__ == "__main__":
    main()
