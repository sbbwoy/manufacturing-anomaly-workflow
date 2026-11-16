"""Evaluate saved validation scores without opening any Test data."""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             fbeta_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

from scripts.validate_ai4i_labels import sha256

TARGET = "Machine failure"
MODELS = {"logistic_regression": "logistic_failure_probability",
          "isolation_forest": "isolation_anomaly_score"}
TRACKING = ["UID", "Product ID", TARGET, "source_row", "csv_row",
            "label_disagreement", "split"]


def validate_scores(data):
    """Validate a complete, unique, finite validation-only score table."""
    required = TRACKING + list(MODELS.values())
    if data.empty or data.columns.duplicated().any():
        raise ValueError("Expected nonempty validation scores with unique columns")
    missing = set(required) - set(data)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if data[required].isna().any().any():
        raise ValueError("Missing validation score or tracking values")
    if not data["split"].eq("validation").all():
        raise ValueError("Only validation rows are allowed")
    if not data[TARGET].isin([0, 1]).all() or data[TARGET].nunique() != 2:
        raise ValueError("Expected both binary target classes")
    if not pd.api.types.is_bool_dtype(data["label_disagreement"]):
        raise ValueError("label_disagreement must be boolean")
    for column in ["UID", "source_row", "csv_row"]:
        if (not pd.api.types.is_numeric_dtype(data[column])
                or not np.isfinite(data[column]).all()
                or not data[column].mod(1).eq(0).all()
                or data[column].duplicated().any()):
            raise ValueError(f"Expected unique integer tracking values: {column}")
    if not (data.source_row.ge(0).all() and data.UID.gt(0).all()
            and data.csv_row.eq(data.source_row + 2).all()):
        raise ValueError("Invalid tracking row positions")
    for column in MODELS.values():
        if (not pd.api.types.is_numeric_dtype(data[column])
                or not np.isfinite(data[column]).all()):
            raise ValueError(f"Expected finite numeric scores: {column}")
    if not data["logistic_failure_probability"].between(0, 1).all():
        raise ValueError("Logistic probabilities must lie within [0, 1]")


def choose_threshold(y, scores):
    """Maximize validation F2 over unique scores; choose highest on exact ties."""
    precision, recall, thresholds = precision_recall_curve(y, scores)
    denominator = 4 * precision[:-1] + recall[:-1]
    f2 = np.divide(5 * precision[:-1] * recall[:-1], denominator,
                   out=np.zeros_like(denominator), where=denominator > 0)
    best = np.flatnonzero(f2 == f2.max())[-1]
    return float(thresholds[best])


def metrics(y, scores, threshold):
    """Compute fixed-threshold and ranking metrics; undefined ranking is null."""
    if len(y) == 0:
        return {"rows": 0, "positives": 0, "threshold": threshold,
                **dict.fromkeys(["ap", "roc_auc", "precision", "recall", "f1", "f2"]),
                "tn": 0, "fp": 0, "fn": 0, "tp": 0}
    predictions = np.asarray(scores) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    both_classes = len(np.unique(y)) == 2
    return {
        "rows": len(y), "positives": int(np.sum(y)), "threshold": threshold,
        "ap": float(average_precision_score(y, scores)) if both_classes else None,
        "roc_auc": float(roc_auc_score(y, scores)) if both_classes else None,
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(fbeta_score(y, predictions, beta=1, zero_division=0)),
        "f2": float(fbeta_score(y, predictions, beta=2, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def run(input_path, output_path):
    """Read one validation CSV and atomically publish metrics, cases and report."""
    input_path, output_path = Path(input_path), Path(output_path)
    if input_path.name != "validation_scores.csv":
        raise ValueError("Input must be named validation_scores.csv")
    if output_path.exists():
        raise FileExistsError(f"Output exists: {output_path}")
    before = sha256(input_path)
    data = pd.read_csv(input_path)
    validate_scores(data)
    all_metrics, sensitivity, case_tables, curves = [], [], {}, {}
    for model, column in MODELS.items():
        y, scores = data[TARGET].to_numpy(), data[column].to_numpy()
        threshold = choose_threshold(y, scores)
        result = {"model": model, **metrics(y, scores, threshold)}
        all_metrics.append(result)
        predicted = scores >= threshold
        cases = data[TRACKING + [column]].copy()
        cases["predicted_failure"] = predicted.astype(int)
        cases["threshold"] = threshold
        cases["error_type"] = np.where(y == 1, "FN", "FP")
        case_tables[model] = cases.loc[predicted != y]
        p, r, t = precision_recall_curve(y, scores)
        curves[model] = pd.DataFrame({"threshold": t, "precision": p[:-1],
                                     "recall": r[:-1]})
        # For binary labels, disagreement with the five-flag OR means its inverse.
        alternative = np.where(data.label_disagreement, 1 - y, y)
        scenarios = [
            ("original", np.ones(len(y), dtype=bool), y),
            ("exclude_disagreements", ~data.label_disagreement.to_numpy(), y),
            ("flag_or_alternative", np.ones(len(y), dtype=bool), alternative),
        ]
        for scenario, mask, labels in scenarios:
            sensitivity.append({"model": model, "scenario": scenario,
                                **metrics(labels[mask], scores[mask], threshold)})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".evaluation-", dir=output_path.parent) as temp:
        staging = Path(temp) / "result"
        staging.mkdir()
        pd.DataFrame(all_metrics).to_csv(staging / "metrics.csv", index=False)
        pd.DataFrame(sensitivity).to_csv(staging / "sensitivity.csv", index=False)
        for model in MODELS:
            case_tables[model].to_csv(staging / f"{model}_errors.csv", index=False)
            curves[model].to_csv(staging / f"{model}_pr_curve.csv", index=False)
        summary = {
            "input": str(input_path), "input_sha256": before,
            "rows": len(data), "positives": int(data[TARGET].sum()),
            "disagreements": int(data.label_disagreement.sum()),
            "score_direction": "higher means more likely failure/anomaly",
            "threshold_rule": "score >= threshold; max validation F2; highest threshold on exact ties",
            "test_read": False, "metrics": all_metrics,
        }
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8")
        lines = [
            "# AI4I Validation 성능 평가",
            "",
            f"Validation {len(data)}행, 고장 {int(data[TARGET].sum())}건, "
            f"라벨 불일치 {int(data.label_disagreement.sum())}건을 평가했다.",
            "입력은 기존 validation_scores.csv 한 파일이며 재학습하지 않았다.",
            "재실행 명령은 python evaluate_validation.py --output artifacts/evaluation_new 이다.",
            "",
            "## 모델 비교와 임계값",
            "",
            "AP는 average_precision_score의 비보간 평균 정밀도이며, 사다리꼴 PR 면적과 구분한다.",
            "임계값은 관측된 고유 점수 중 F2 최대값을 선택했다. 동률이면 높은 임계값을 선택한다.",
            "점수 ≥ 임계값이면 고장으로 분류한다. F2는 Recall에 더 큰 가중치를 둔다.",
            "",
        ]
        for row in all_metrics:
            lines += [
                f"- {row['model']} — AP {row['ap']:.6f}, ROC-AUC {row['roc_auc']:.6f}, "
                f"임계값 {row['threshold']:.12g}, Precision {row['precision']:.6f}, "
                f"Recall {row['recall']:.6f}, F1 {row['f1']:.6f}, F2 {row['f2']:.6f}. "
                f"TN/FP/FN/TP = {row['tn']}/{row['fp']}/{row['fn']}/{row['tp']}."
            ]
        lines += ["", "## 라벨 민감도", "",
                  "임계값을 고정하고 원본 라벨, 불일치 관측 제외, 플래그 OR 대체 라벨을 비교했다.",
                  "OR 대체 라벨은 기존 감사 불일치 여부를 이용해 해당 이진 라벨만 반전했다.",
                  "원본 라벨과 행은 수정하지 않았다. 대체 라벨의 정답 여부는 미확인이다.", ""]
        for row in sensitivity:
            if row["rows"] == 0:
                lines.append(f"- {row['model']} / {row['scenario']} — 0행, 지표 산출 불가.")
                continue
            lines.append(f"- {row['model']} / {row['scenario']} — "
                         f"{row['rows']}행, 양성 {row['positives']}건, "
                         f"Precision {row['precision']:.6f}, Recall {row['recall']:.6f}, "
                         f"F2 {row['f2']:.6f}, FP/FN {row['fp']}/{row['fn']}.")
        lines += ["", "## 저장 결과와 한계", "",
                  "- metrics.csv에 AP, ROC-AUC, 임계값 지표와 혼동행렬을 저장했다.",
                  "- 모델별 errors.csv에 UID, 실제 라벨, 점수, FP/FN 및 불일치 여부를 저장했다.",
                  "- sensitivity.csv에 각 시나리오의 전체 지표를 저장했다.",
                  "- 모델별 pr_curve.csv에는 임계값별 Precision/Recall을 저장했다.",
                  "- 같은 Validation에서 임계값을 선택하고 평가했으므로 성능에 선택 편향이 있다.",
                  "- 기존 전체 데이터 EDA와 무작위 분할의 시간·설비 독립성 한계가 남는다.",
                  "- 고장 라벨은 원본 Machine failure이며 원인이나 실제 운영 성능을 단정할 수 없다.",
                  "- Test 데이터는 이번 실행에서 읽거나 평가하지 않았다. 최종 평가가 후속 과제다."]
        (staging / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        if sha256(input_path) != before:
            raise RuntimeError("Validation input changed")
        staging.rename(output_path)
    return summary


def main():
    """Evaluate the saved validation split only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path("artifacts/modeling/validation_scores.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation"))
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output)["metrics"], indent=2))


if __name__ == "__main__":
    main()
