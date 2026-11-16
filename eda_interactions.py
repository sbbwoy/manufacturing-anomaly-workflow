"""Additional AI4I EDA. Run python eda_interactions.py from the project root."""
import argparse
from hashlib import sha256
from pathlib import Path

from eda import MODES, PROCESS, TARGET, load_data, save_figure, pd, plt, sns

SPEED, TORQUE, WEAR = PROCESS[2:]
GAP = "Temperature difference [K]"
LOAD = "Torque x tool wear [Nm min]"


def scatter(ax, data, x, y, label):
    """Plot every record, drawing flagged observations last for visibility."""
    for flag, color in [(0, "grey"), (1, "crimson")]:
        subset = data.loc[data[label].eq(flag)]
        ax.scatter(subset[x], subset[y], s=9 if flag == 0 else 18,
                   alpha=0.15 if flag == 0 else 0.8, color=color,
                   label=f"{label}={flag} (n={len(subset)})")
    ax.set(xlabel=x, ylabel=y)
    ax.legend(fontsize=8)


def main():
    """Validate input, save interaction plots and tables, and report observations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/ai4i2020.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/eda_interactions"))
    args = parser.parse_args()
    # A new directory prevents overwriting the original EDA or another run.
    if args.output.exists():
        parser.error("Output directory already exists. Choose a new --output directory.")
    original_hash = sha256(args.input.read_bytes()).hexdigest()
    data = load_data(args.input).copy()
    if data[PROCESS].isna().any().any():
        raise ValueError("Process variables must be nonmissing for interaction analysis")
    if not data["Type"].isin(["L", "M", "H"]).all():
        raise ValueError("Expected product types L, M, H")
    data[GAP] = data[PROCESS[1]] - data[PROCESS[0]]
    data[LOAD] = data[TORQUE] * data[WEAR]
    args.output.mkdir(parents=True)
    sns.set_theme()
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    for ax, label in zip(axes.flat, [TARGET] + MODES):
        scatter(ax, data, SPEED, TORQUE, label)
    save_figure(fig, args.output, "speed_torque.png")
    rows = []
    for label in [TARGET] + MODES:
        for value, group in data.groupby(label):
            rows.append({"label": label, "value": value, "count": len(group),
                         "speed_median": group[SPEED].median(),
                         "torque_median": group[TORQUE].median(),
                         "pearson": group[SPEED].corr(group[TORQUE])})
    stats = pd.DataFrame(rows)
    stats.to_csv(args.output / "speed_torque_groups.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    scatter(axes[0], data, GAP, SPEED, "HDF")
    sns.boxplot(data=data, x="HDF", y=GAP, ax=axes[1])
    save_figure(fig, args.output, "temperature_hdf.png")
    gap_stats = data.groupby("HDF")[[GAP, SPEED]].agg(
        ["count", "min", "median", "max"])
    gap_stats.to_csv(args.output / "hdf_groups.csv")
    # Fixed-width descriptive bins, not fitted decision boundaries.
    bins = pd.cut(data[GAP], bins=range(0, 22, 2), right=False)
    binned = data.groupby(bins, observed=True)["HDF"].agg(["count", "sum", "mean"])
    binned.to_csv(args.output / "hdf_gap_bins.csv")
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharex=True, sharey=True)
    for ax, product_type in zip(axes, ["L", "M", "H"]):
        scatter(ax, data.loc[data["Type"].eq(product_type)], WEAR, TORQUE, "OSF")
        ax.set_title(f"Type {product_type}")
    save_figure(fig, args.output, "wear_torque_osf.png")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=data, x="Type", y=LOAD, hue="OSF",
                order=["L", "M", "H"], ax=ax)
    save_figure(fig, args.output, "load_osf.png")
    load_stats = data.groupby(["Type", "OSF"])[LOAD].agg(
        ["count", "min", "median", "max"])
    load_stats.to_csv(args.output / "osf_load_groups.csv")
    type_rates = data.groupby("Type")["OSF"].agg(["count", "sum", "mean"])
    type_rates.to_csv(args.output / "osf_type_rates.csv")
    lines = []
    for label in [TARGET] + MODES:
        group = stats.loc[stats["label"].eq(label) & stats["value"].eq(1)].iloc[0]
        lines.append(f"- {label}=1은 {int(group['count'])}건, 회전 속도 중앙값 "
                     f"{group['speed_median']:g} rpm, 토크 중앙값 {group['torque_median']:g} Nm.")
    hdf_lines = []
    for flag, row in gap_stats.iterrows():
        hdf_lines.append(f"- HDF={flag}의 온도 차이 최소/중앙/최대는 "
                         f"{row[(GAP, 'min')]:.2f}/{row[(GAP, 'median')]:.2f}/"
                         f"{row[(GAP, 'max')]:.2f} K.")
    osf_lines = []
    for kind, row in type_rates.iterrows():
        flagged = load_stats.loc[(kind, 1)]
        osf_lines.append(f"- Type {kind}의 OSF는 {int(row['sum'])}/{int(row['count'])}건"
                         f"({row['mean']:.2%}), OSF=1 부하 곱 최소/중앙/최대는 "
                         f"{flagged['min']:g}/{flagged['median']:g}/{flagged['max']:g} Nm·min.")
    report = f"""# AI4I 공정 변수 조합 추가 EDA

입력 SHA-256은 `{original_hash}`이다. 원본 열을 보존하고 메모리에서 파생변수를 계산했다.
기존 EDA의 입력 검증과 그래프 저장 함수를 재사용했다.

## 회전 속도와 토크

관찰 사실
{chr(10).join(lines)}

[산점도](speed_torque.png)의 각 패널은 해당 플래그를 독립적으로 구분한다.
한 행에 여러 유형이 있으면 여러 패널에 표시된다.
전체 데이터의 Pearson 상관은 {data[SPEED].corr(data[TORQUE]):.3f}이다.

해석과 한계 — 두 변수의 음의 연관을 관찰했다. 그룹별 중앙값과 분포를 함께 검토해야 하며,
한 변수나 평균적 경향만으로 모든 고장을 구분한다고 해석할 수 없다.
유형별 표본 수와 중복 플래그가 다르고, 점의 겹침이 밀도를 가릴 수 있다.

## 공정 온도 − 공기 온도와 HDF

파생변수는 공정 온도에서 공기 온도를 뺀 값이며 단위는 K이다.
관찰 사실
{chr(10).join(hdf_lines)}

[온도 차이와 HDF](temperature_hdf.png) 및 [2 K 구간별 빈도와 HDF 비율](hdf_gap_bins.csv)을 저장했다.
해석과 한계 — HDF별 온도 차이와 회전 속도의 분포를 함께 살펴볼 수 있다.
온도 차이만으로 HDF를 설명하거나 인과관계를 추론할 수 없다.
구간 집계는 탐색용이며 구간 설정에 따라 요약이 달라진다.

## 토크 × 공구 마모와 OSF

파생변수는 토크와 공구 마모의 곱이며 단위는 Nm·min이다.
관찰 사실
{chr(10).join(osf_lines)}

[유형별 토크와 마모](wear_torque_osf.png), [부하 곱 분포](load_osf.png),
[OSF별 제품 유형 통계](osf_load_groups.csv)를 저장했다.
해석과 한계 — 제품 유형별 빈도와 부하 곱 분포를 함께 비교했다.
유형별 표본 수와 운전 조건의 차이가 있어 고장률 차이를 제품 유형의 인과 효과로 볼 수 없다.

## 공통 한계와 후속 검증

- 합성 데이터의 관측적 EDA이다. 원인 규명이나 실제 설비의 성능 검증을 수행하지 않았다.
- 임계값을 학습하거나 모델을 평가하지 않았다. 후속 검증에서 라벨 생성 규칙과 측정 정밀도를 확인한다.
- 고장 유형과 Machine failure의 기존 불일치 라벨을 그대로 유지했다.
- TWF/HDF/PWF/OSF/RNF와 이 플래그에서 만든 파생값은 향후 예측 입력에서 제외한다.
  UID와 Product ID도 식별자로 제외한다. 이번 두 공정 파생변수는 예측 시점의 가용성 검증이 필요하다.
"""
    (args.output / "report.md").write_text(report, encoding="utf-8")
    if sha256(args.input.read_bytes()).hexdigest() != original_hash:
        raise RuntimeError("Input changed during analysis")


if __name__ == "__main__":
    main()
