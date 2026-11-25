"""Compare unsupervised anomaly models on frozen Train/Validation inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import sklearn
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, fbeta_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from train_models import FEATURES, TARGET, TRACK_COLS

SPLITS = ("train", "validation")
TRACK_META_COLUMNS = [column for column in TRACK_COLS if column != TARGET]
SCORE_COLUMNS = {
    "isolation_forest": "isolation_forest_anomaly_score",
    "lof": "lof_anomaly_score",
    "one_class_svm": "one_class_svm_anomaly_score",
    "pca": "pca_reconstruction_error",
}
MODEL_SETTINGS = {
    "isolation_forest": {
        "n_estimators": 200, "contamination": "auto", "random_state": 42,
        "n_jobs": 1,
    },
    "lof": {"n_neighbors": 20, "novelty": True, "contamination": "auto"},
    "one_class_svm": {"nu": 0.05, "kernel": "rbf", "gamma": "scale"},
    "pca": {"n_components": 0.95, "svd_solver": "full"},
}
THRESHOLD_QUANTILE = 0.95


def _sha256(path: Path) -> str:
    """Return a file's SHA-256 digest without loading it into memory at once."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_paths(input_dir: Path) -> dict[str, Path]:
    """Return the only files this experiment is allowed to read or hash."""
    return {
        f"{split}_{kind}.csv": input_dir / f"{split}_{kind}.csv"
        for split in SPLITS
        for kind in ("features", "tracking")
    }


def _load_features(path: Path, split: str) -> pd.DataFrame:
    """Load one numeric split using the frozen 11-column feature schema."""
    features = pd.read_csv(path)
    if features.empty or list(features.columns) != FEATURES:
        raise ValueError(f"{split}: missing rows or unexpected feature schema")
    if not all(pd.api.types.is_numeric_dtype(features[column]) for column in FEATURES):
        raise ValueError(f"{split}: feature data must be numeric")
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError(f"{split}: features contain missing or nonfinite values")
    return features


def _load_tracking_metadata(path: Path, split: str, rows: int) -> pd.DataFrame:
    """Load tracking columns while deliberately excluding Machine failure."""
    header = pd.read_csv(path, nrows=0)
    if list(header.columns) != TRACK_COLS:
        raise ValueError(f"{split}: unexpected tracking schema")
    metadata = pd.read_csv(path, usecols=TRACK_META_COLUMNS)[TRACK_META_COLUMNS]
    if len(metadata) != rows or metadata[TRACK_META_COLUMNS].isna().any().any():
        raise ValueError(f"{split}: missing tracking values or row count mismatch")
    if metadata["split"].ne(split).any():
        raise ValueError(f"{split}: tracking contains a different split")
    if not pd.api.types.is_bool_dtype(metadata["label_disagreement"]):
        raise ValueError(f"{split}: label_disagreement must be boolean")
    for column in ("UID", "source_row", "csv_row"):
        values = metadata[column]
        if (not pd.api.types.is_numeric_dtype(values)
                or not np.isfinite(values).all()
                or not values.mod(1).eq(0).all()
                or values.duplicated().any()):
            raise ValueError(f"{split}: invalid or duplicate {column}")
    if (not metadata["UID"].gt(0).all() or not metadata["source_row"].ge(0).all()
            or not metadata["csv_row"].eq(metadata["source_row"] + 2).all()):
        raise ValueError(f"{split}: invalid tracking row positions")
    return metadata


def _load_validation_labels(path: Path, rows: int) -> np.ndarray:
    """Read validation labels for evaluation only, after every score is computed."""
    labels = pd.read_csv(path, usecols=[TARGET])[TARGET]
    if len(labels) != rows or labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("validation: expected one binary Machine failure label per row")
    result = labels.to_numpy(dtype=int)
    if np.unique(result).size != 2:
        raise ValueError("validation: AP and ROC-AUC require both label classes")
    return result


def _build_models() -> dict[str, object]:
    """Create fresh estimators with the fixed comparison settings."""
    return {
        "isolation_forest": IsolationForest(**MODEL_SETTINGS["isolation_forest"]),
        "lof": LocalOutlierFactor(**MODEL_SETTINGS["lof"]),
        "one_class_svm": OneClassSVM(**MODEL_SETTINGS["one_class_svm"]),
        "pca": PCA(**MODEL_SETTINGS["pca"]),
    }


def _score_model(name: str, model: object, features: np.ndarray) -> np.ndarray:
    """Return anomaly scores where larger values always indicate more anomaly."""
    matrix = np.asarray(features, dtype=float)
    if name == "isolation_forest":
        scores = -model.decision_function(matrix)
    elif name == "lof":
        scores = -model.score_samples(matrix)
    elif name == "one_class_svm":
        scores = -model.decision_function(matrix)
    elif name == "pca":
        reconstructed = model.inverse_transform(model.transform(matrix))
        scores = np.mean(np.square(matrix - reconstructed), axis=1)
    else:  # pragma: no cover - guarded by the fixed model list
        raise ValueError(f"Unknown model: {name}")
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(features),) or not np.isfinite(scores).all():
        raise ValueError(f"{name}: produced invalid validation scores")
    return scores


def _fit_and_score(train: pd.DataFrame, validation: pd.DataFrame):
    """Fit on Train features and score Validation before labels are loaded."""
    models = _build_models()
    fit_seconds, score_seconds, scores = {}, {}, {}
    train_matrix = train.to_numpy(dtype=float)
    validation_matrix = validation.to_numpy(dtype=float)
    for name, model in models.items():
        start = time.perf_counter()
        model.fit(train_matrix)
        fit_seconds[name] = time.perf_counter() - start
        start = time.perf_counter()
        scores[name] = _score_model(name, model, validation_matrix)
        score_seconds[name] = time.perf_counter() - start
    return models, scores, fit_seconds, score_seconds


def _evaluate_model(labels: np.ndarray, scores: np.ndarray) -> tuple[dict, pd.DataFrame]:
    """Calculate ranking metrics and fixed 95th-percentile threshold metrics."""
    threshold = float(np.quantile(scores, THRESHOLD_QUANTILE))
    predicted = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    curve = pd.DataFrame({
        "threshold": np.r_[thresholds, np.nan],
        "precision": precision,
        "recall": recall,
    })
    return ({
        "rows": int(len(labels)),
        "positives": int(labels.sum()),
        "ap": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "threshold_quantile": THRESHOLD_QUANTILE,
        "threshold": threshold,
        "flagged_rows": int(predicted.sum()),
        "flagged_fraction": float(predicted.mean()),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "f2": float(fbeta_score(labels, predicted, beta=2, zero_division=0)),
        "fpr": float(fp / (fp + tn)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }, curve)


def _plot_pr_curves(curves: dict[str, pd.DataFrame], metrics: dict[str, dict], path: Path) -> None:
    """Save a shared Precision-Recall curve figure for the four models."""
    figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for name, curve in curves.items():
        result = metrics[name]
        axis.plot(curve["recall"], curve["precision"],
                  label=f"{name} (AP={result['ap']:.3f})")
    axis.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02),
             title="Validation Precision-Recall Curves")
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run(input_dir: Path, output_dir: Path) -> dict:
    """Fit, score and compare four unsupervised models without accessing Test."""
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Output path already exists: {output_dir}")
    paths = _input_paths(input_dir)
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Train/Validation inputs: {missing}")
    input_hashes = {name: _sha256(path) for name, path in paths.items()}

    train = _load_features(paths["train_features.csv"], "train")
    validation = _load_features(paths["validation_features.csv"], "validation")
    train_meta = _load_tracking_metadata(paths["train_tracking.csv"], "train", len(train))
    validation_meta = _load_tracking_metadata(
        paths["validation_tracking.csv"], "validation", len(validation))
    if (set(train_meta["source_row"]) & set(validation_meta["source_row"])
            or set(train_meta["UID"]) & set(validation_meta["UID"])):
        raise ValueError("Train and Validation tracking overlap")

    models, scores, fit_seconds, score_seconds = _fit_and_score(train, validation)
    # This is the first point where target values are parsed or made available.
    labels = _load_validation_labels(paths["validation_tracking.csv"], len(validation))

    metrics_by_model, curves = {}, {}
    metric_rows = []
    for name in models:
        result, curve = _evaluate_model(labels, scores[name])
        metrics_by_model[name] = {
            **result,
            "fit_time_seconds": float(fit_seconds[name]),
            "score_time_seconds": float(score_seconds[name]),
            "total_model_time_seconds": float(fit_seconds[name] + score_seconds[name]),
        }
        curves[name] = curve.assign(model=name)
        metric_rows.append({"model": name, **metrics_by_model[name]})

    # AP is the primary selection metric. Exact ties use ROC-AUC, Recall, FPR,
    # then combined fit/score time in that order.
    selected = max(models, key=lambda name: (
        metrics_by_model[name]["ap"], metrics_by_model[name]["roc_auc"],
        metrics_by_model[name]["recall"], -metrics_by_model[name]["fpr"],
        -metrics_by_model[name]["total_model_time_seconds"],
    ))
    for row in metric_rows:
        row["selected"] = row["model"] == selected
    metric_frame = pd.DataFrame(metric_rows)
    comparison_frame = metric_frame.sort_values(
        ["ap", "roc_auc", "recall", "fpr", "total_model_time_seconds"],
        ascending=[False, False, False, True, True], kind="stable").reset_index(drop=True)
    comparison_frame.insert(0, "rank", np.arange(1, len(comparison_frame) + 1))
    curve_frame = pd.concat(curves.values(), ignore_index=True)
    score_frame = validation_meta.copy()
    score_frame[TARGET] = labels
    for name, column in SCORE_COLUMNS.items():
        score_frame[column] = scores[name]

    settings = {
        "feature_names": FEATURES,
        "feature_count": len(FEATURES),
        "training_split": "train",
        "scored_split": "validation",
        "fit_uses_labels": False,
        "validation_labels_loaded_after_scoring": True,
        "test_touched": False,
        "score_direction": "higher means more anomalous",
        "threshold_rule": (
            "Per-model validation anomaly-score 95th percentile; flag score >= threshold"
        ),
        "threshold_quantile": THRESHOLD_QUANTILE,
        "threshold_assumption": (
            "Temporary unlabeled 5% candidate-review assumption; ties at the quantile "
            "can flag more than 5%. Not optimized using labels."
        ),
        "models": MODEL_SETTINGS,
        "sklearn_version": sklearn.__version__,
        "input_sha256": input_hashes,
        "training_rows": len(train),
        "validation_rows": len(validation),
        "validation_positives": int(labels.sum()),
        "selection_metric": "AP (average precision)",
        "selected_model": selected,
        "results": metrics_by_model,
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".unsupervised-", dir=output_dir.parent) as temp:
        staging = Path(temp) / "result"
        staging.mkdir()
        for name, model in models.items():
            joblib.dump(model, staging / f"{name}.joblib")
        score_frame.to_csv(staging / "validation_scores.csv", index=False)
        metric_frame.to_csv(staging / "metrics.csv", index=False)
        comparison_frame.to_csv(staging / "comparison.csv", index=False)
        curve_frame.to_csv(staging / "pr_curves.csv", index=False)
        _plot_pr_curves(curves, metrics_by_model, staging / "precision_recall_curves.png")
        (staging / "summary.json").write_text(
            json.dumps(settings, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8")
        (staging / "report.md").write_text(
            _build_report(settings, comparison_frame), encoding="utf-8")
        if any(_sha256(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("Train/Validation preprocessing inputs changed during comparison")
        staging.rename(output_dir)
    return settings


def _build_report(settings: dict, comparison: pd.DataFrame) -> str:
    """Build the human-readable comparison and selection report."""
    lines = [
        "# 비지도 이상탐지 모델 비교",
        "",
        f"Train {settings['training_rows']:,}행의 11개 특징으로 학습하고, "
        f"Validation {settings['validation_rows']:,}행에서 점수를 산출했다. "
        f"Validation Machine failure 양성은 {settings['validation_positives']:,}건이다.",
        "학습에는 특징값만 전달했다. 모든 Validation 점수를 산출한 뒤에만 라벨을 읽어 평가했다. "
        "Test 입력·라벨·매니페스트는 읽거나 평가하지 않았다.",
        "",
        "## 비교 결과",
        "",
        "모델은 AP 내림차순으로 정렬했다. 임계값은 모델별 Validation 점수의 95분위수이며, "
        "라벨 기반으로 조정하지 않았다.",
        "",
        "|순위|모델|AP|ROC-AUC|Recall|FPR|Precision|F1|F2|표시 비율|학습 초|점수 초|선정|",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for _, row in comparison.iterrows():
        lines.append(
            f"|{int(row['rank'])}|{row['model']}|{row['ap']:.6f}|{row['roc_auc']:.6f}|"
            f"{row['recall']:.6f}|{row['fpr']:.6f}|{row['precision']:.6f}|"
            f"{row['f1']:.6f}|{row['f2']:.6f}|{row['flagged_fraction']:.3%}|"
            f"{row['fit_time_seconds']:.6f}|{row['score_time_seconds']:.6f}|"
            f"{'예' if row['selected'] else ''}|"
        )
    selected = settings["selected_model"]
    selected_row = comparison.loc[comparison.model.eq(selected)].iloc[0]
    lines += [
        "",
        "## 최종 후보",
        "",
        f"**{selected}**를 후속 검토 후보로 선정했다. AP {selected_row['ap']:.6f}가 네 모델 중 "
        "가장 높았다. AP를 주 지표로 두고 ROC-AUC, Recall, FPR 및 학습·점수 산출시간을 "
        "보조 지표로 확인했다. AP가 같으면 ROC-AUC, Recall, 낮은 FPR, 짧은 합산 실행시간 순으로 "
        "결정한다. 이는 후보 선정이며 운영 적용 또는 고장 원인 판정을 의미하지 않는다.",
        "",
        "## 점수와 임계값",
        "",
        "- Isolation Forest: `-decision_function`.",
        "- LOF: `novelty=True`로 학습하고 Validation 신규 관측에 `-score_samples` 적용.",
        "- One-Class SVM: `-decision_function`.",
        "- PCA: 재구성 평균제곱오차(mean squared reconstruction error).",
        "- 모든 점수는 클수록 이상에 가깝다. PR 곡선은 `pr_curves.csv`와 "
        "`precision_recall_curves.png`에 저장했다.",
        "- 95분위수 임계값은 라벨 없는 임시 5% 후보 검토 가정이다. 임계값 점수 동률로 "
        "실제 표시 비율은 5%보다 높을 수 있다.",
        "",
        "## 설정과 산출물",
        "",
        "- 모델별 `.joblib`, `validation_scores.csv`, `metrics.csv`, `comparison.csv`, "
        "PR 곡선, `summary.json` 및 이 보고서를 저장했다.",
        "- `summary.json`에 입력 파일 SHA-256, 설정, scikit-learn 버전, 학습·점수 시간과 "
        "결과 요약을 기록했다. 해시는 Train/Validation 특징 및 추적 CSV만 포함한다.",
        "- 재실행: `python compare_unsupervised.py --input artifacts/preprocessing "
        "--output artifacts/unsupervised_comparison_new`",
        "",
        "## 한계",
        "",
        "Machine failure 라벨은 평가용 대리 라벨일 뿐 이상 원인이나 모든 유형의 공정 이상을 "
        "뜻하지 않는다. 라벨 불일치와 무작위 분할의 시간·설비 독립성은 기존 한계로 남는다. "
        "Validation으로 모델을 선정했으므로 독립 Test 평가 전의 후보 결과다.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("artifacts/preprocessing"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/unsupervised_comparison"))
    args = parser.parse_args()
    summary = run(args.input, args.output)
    print(json.dumps({
        "selected_model": summary["selected_model"],
        "validation_rows": summary["validation_rows"],
        "test_touched": summary["test_touched"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
