"""One-time final holdout inference using a frozen validation-selected model."""
import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from evaluate_validation import metrics
from scripts.validate_ai4i_labels import sha256
from train_models import FEATURES, TARGET, TRACK_COLS, load_split

THRESHOLD = 0.7218245828133553
MODEL_PATH = "artifacts/modeling/logistic_regression.joblib"
PINNED_PATHS = [MODEL_PATH, "artifacts/modeling/summary.json",
                "artifacts/evaluation/metrics.csv", "artifacts/preprocessing/summary.json"]


def update_record(path, record):
    """Atomically replace an already claimed execution record."""
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                     dir=path.parent, delete=False) as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2, allow_nan=False)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def check_manifest(manifest):
    """Reject ambiguous tracking or split overlap before inference."""
    if list(manifest.columns) != TRACK_COLS or manifest.isna().any().any():
        raise ValueError("Invalid manifest schema or missing values")
    if not manifest["split"].isin(["train", "validation", "test"]).all():
        raise ValueError("Unknown manifest split")
    for column in ["UID", "source_row", "csv_row"]:
        if (not pd.api.types.is_numeric_dtype(manifest[column])
                or not np.isfinite(manifest[column]).all()
                or not manifest[column].mod(1).eq(0).all()
                or manifest[column].duplicated().any()):
            raise ValueError("Duplicate or invalid tracking identifiers")
    if not (manifest.source_row.ge(0).all() and manifest.UID.gt(0).all()
            and manifest.csv_row.eq(manifest.source_row + 2).all()):
        raise ValueError("Invalid source positions")
    if (not manifest[TARGET].isin([0, 1]).all()
            or not pd.api.types.is_bool_dtype(manifest.label_disagreement)):
        raise ValueError("Invalid manifest labels or audit flags")


def run(config_path):
    """Claim a persistent one-shot ledger before Test reads; never fit or retune."""
    config_path = Path(config_path).resolve()
    if config_path.name != "final_holdout.json" or config_path.parent.name != "config":
        raise ValueError("Expected config/final_holdout.json")
    root = config_path.parent.parent
    output = root / "artifacts/final_holdout"
    ledger = root / "artifacts/final_holdout_execution.json"
    if ledger.exists() or output.exists():
        raise FileExistsError("Holdout already claimed or evaluated; automatic rerun forbidden")
    config_hash = sha256(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (config.get("model") != "logistic_regression"
            or config.get("threshold") != THRESHOLD
            or config.get("prediction_rule") != "score >= threshold"
            or config.get("selection_reason") !=
                "Highest Validation AP; threshold maximizes Validation F2"
            or not isinstance(config.get("expected_rows"), int)
            or config["expected_rows"] < 1):
        raise ValueError("Frozen selection policy does not match approved model/threshold")
    if set(config.get("pinned_sha256", {})) != set(PINNED_PATHS):
        raise ValueError("Missing frozen input hashes")
    for relative, expected in config["pinned_sha256"].items():
        if sha256(root / relative) != expected:
            raise ValueError(f"Frozen input hash mismatch: {relative}")
    preproc = root / "artifacts/preprocessing"
    metadata = json.loads((preproc / "summary.json").read_text(encoding="utf-8"))
    if metadata.get("feature_names") != FEATURES or metadata.get("target") != TARGET:
        raise ValueError("Preprocessing metadata schema mismatch")
    validation = pd.read_csv(root / "artifacts/evaluation/metrics.csv")
    chosen = validation.loc[validation.model.eq("logistic_regression")]
    if (len(chosen) != 1 or chosen.iloc[0].ap != validation.ap.max()
            or not np.isclose(chosen.iloc[0].threshold, THRESHOLD, rtol=0, atol=1e-15)):
        raise ValueError("Validation selection evidence mismatch")
    model = joblib.load(root / MODEL_PATH)
    if (not isinstance(model, LogisticRegression)
            or list(model.feature_names_in_) != FEATURES
            or list(model.classes_) != [0, 1]):
        raise ValueError("Saved model type, classes or feature schema mismatch")
    if sha256(config_path) != config_hash or any(
            sha256(root / relative) != expected
            for relative, expected in config["pinned_sha256"].items()):
        raise ValueError("Frozen configuration or model changed during preflight")
    record = {
        "status": "claimed", "config_sha256": config_hash,
        "model": config["model"], "threshold": THRESHOLD,
        "claimed_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "One-shot claim persists on failure. Do not delete or reset to rerun.",
    }
    ledger.parent.mkdir(parents=True, exist_ok=True)
    # O_EXCL creation rejects concurrent runners before either can open Test.
    with ledger.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    try:
        protected = [config_path] + list((root / "data").rglob("*.csv")) + [
            p for p in (root / "artifacts").rglob("*") if p.is_file() and p != ledger]
        before = {str(p.relative_to(root)): sha256(p) for p in protected}
        raw = root / "data/ai4i2020.csv"
        if raw.exists() and sha256(raw) != metadata["input_sha256"]:
            raise ValueError("Original data provenance hash mismatch")
        manifest = pd.read_csv(preproc / "split_manifest.csv")
        check_manifest(manifest)
        features, tracking = load_split(preproc, "test", FEATURES, manifest)
        if len(features) != config["expected_rows"]:
            raise ValueError("Unexpected Test row count")
        if (tracking[TARGET].nunique() != 2
                or not pd.api.types.is_bool_dtype(tracking.label_disagreement)):
            raise ValueError("Expected both Test classes and boolean audit flags")
        # Exactly one predict_proba call. No fit, model selection or threshold search.
        scores = model.predict_proba(features)[:, 1]
        if not np.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
            raise ValueError("Invalid model probabilities")
        result = metrics(tracking[TARGET].to_numpy(), scores, THRESHOLD)
        predicted = (scores >= THRESHOLD).astype(int)
        predictions = tracking.copy()
        predictions["logistic_failure_probability"] = scores
        predictions["predicted_failure"] = predicted
        predictions["threshold"] = THRESHOLD
        wrong = predicted != tracking[TARGET].to_numpy()
        errors = predictions.loc[wrong].copy()
        errors["error_type"] = np.where(errors[TARGET].eq(1), "FN", "FP")
        after = {p: sha256(root / p) for p in before}
        if before != after:
            raise RuntimeError("Frozen input or existing artifacts changed")
        with tempfile.TemporaryDirectory(prefix=".holdout-", dir=output.parent) as temp:
            staging = Path(temp) / "result"
            staging.mkdir()
            predictions.to_csv(staging / "test_predictions.csv", index=False)
            errors.to_csv(staging / "test_errors.csv", index=False)
            pd.DataFrame([{"model": "logistic_regression", **result}]).to_csv(
                staging / "metrics.csv", index=False)
            pd.DataFrame([[result["tn"], result["fp"]], [result["fn"], result["tp"]]],
                         index=["actual_0", "actual_1"],
                         columns=["predicted_0", "predicted_1"]).to_csv(
                             staging / "confusion_matrix.csv")
            summary = {
                "config_sha256": config_hash, "model": "logistic_regression",
                "metrics": result, "label_disagreements": int(tracking.label_disagreement.sum()),
                "predict_proba_calls": 1, "fit_calls": 0,
                "threshold_search_performed": False,
                "input_hashes": before, "inputs_unchanged": True,
            }
            (staging / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
                encoding="utf-8")
            report = f"""# AI4I 최종 Test 평가

Validation AP 최고 모델 Logistic Regression과 Validation F2 최대화 임계값
{THRESHOLD}을 Test 접근 전에 config/final_holdout.json에 고정했다.
설정 SHA-256은 `{config_hash}`이다.

## 실제 실행 결과

- Test {result['rows']}건, 원본 Machine failure 양성 {result['positives']}건
- AP(비보간 평균 정밀도) {result['ap']:.6f}, ROC-AUC {result['roc_auc']:.6f}
- Precision {result['precision']:.6f}, Recall {result['recall']:.6f}
- F1 {result['f1']:.6f}, F2 {result['f2']:.6f}
- 혼동행렬 TN/FP/FN/TP는 {result['tn']}/{result['fp']}/{result['fn']}/{result['tp']}
- 라벨 불일치 관측 {summary['label_disagreements']}건을 보존했다.
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
"""
            (staging / "report.md").write_text(report, encoding="utf-8")
            staging.rename(output)
        record.update(status="completed", completed_at_utc=datetime.now(timezone.utc).isoformat(),
                      output="artifacts/final_holdout", predict_proba_calls=1)
        update_record(ledger, record)
        return summary
    except Exception as error:
        record.update(status="failed", error_type=type(error).__name__,
                      automatic_retry_allowed=False)
        update_record(ledger, record)
        raise


def main():
    """Execute the frozen evaluation once. No override or reset options exist."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/final_holdout.json"))
    args = parser.parse_args()
    print(json.dumps(run(args.config)["metrics"], indent=2))


if __name__ == "__main__":
    main()
