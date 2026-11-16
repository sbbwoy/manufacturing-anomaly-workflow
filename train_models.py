"""Fit baseline AI4I models on frozen preprocessing splits (test set untouched)."""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression

SEED = 42
SPLITS = ("train", "validation")
TARGET = "Machine failure"
# Keep this explicit to reject accidental target, mode or identifier leakage.
FEATURES = [
    "type__Type_L", "type__Type_M", "type__Type_H",
    "numeric__Air temperature [K]", "numeric__Process temperature [K]",
    "numeric__Rotational speed [rpm]", "numeric__Torque [Nm]",
    "numeric__Tool wear [min]", "numeric__Temperature difference [K]",
    "numeric__Mechanical power [W]", "numeric__Torque x tool wear [Nm min]",
]
TRACK_COLS = ["UID", "Product ID", TARGET, "source_row", "csv_row", "label_disagreement", "split"]


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split(input_dir: Path, name: str, expected: list[str], manifest: pd.DataFrame):
    """Read feature/tracking files; check shape, schema, source-order manifest."""
    features = pd.read_csv(input_dir / f"{name}_features.csv")
    tracking = pd.read_csv(input_dir / f"{name}_tracking.csv")
    if features.empty or list(features.columns) != expected:
        raise ValueError(f"{name}: missing rows or unexpected feature schema")
    if not all(pd.api.types.is_numeric_dtype(features[col]) for col in features):
        raise ValueError(f"{name}: feature data must be numeric")
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError(f"{name}: features contain missing/nonfinite values")
    if list(tracking.columns) != TRACK_COLS or len(features) != len(tracking):
        raise ValueError(f"{name}: tracking schema or row count mismatch")
    if tracking[TRACK_COLS].isna().any().any() or not tracking[TARGET].isin([0, 1]).all():
        raise ValueError(f"{name}: invalid labels or missing tracking values")
    if tracking["split"].ne(name).any() or tracking["source_row"].duplicated().any():
        raise ValueError(f"{name}: incorrect split or duplicate source_row")
    original = manifest.loc[manifest["split"] == name].set_index("source_row")
    actual = tracking.set_index("source_row")
    if len(original) != len(actual) or not original.index.is_unique:
        raise ValueError(f"{name}: split manifest coverage mismatch")
    columns = ["UID", "Product ID", TARGET, "csv_row", "label_disagreement", "split"]
    if not original.reindex(actual.index)[columns].equals(actual[columns]):
        raise ValueError(f"{name}: tracking disagrees with split manifest")
    return features, tracking


def run(input_dir: Path, output_dir: Path) -> dict:
    """Train models only on Train; save scores for Train/Validation only."""
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Output path already exists: {output_dir}")
    config_file = input_dir / "summary.json"
    config = json.loads(config_file.read_text(encoding="utf-8"))
    if config.get("target") != TARGET or config.get("feature_names") != FEATURES:
        raise ValueError("Unexpected preprocessing target or feature names")
    manifest = pd.read_csv(input_dir / "split_manifest.csv")
    if not set(TRACK_COLS).issubset(manifest.columns):
        raise ValueError("Invalid split manifest")
    data = {name: load_split(input_dir, name, FEATURES, manifest) for name in SPLITS}
    ids_train = set(data["train"][1]["source_row"])
    ids_val = set(data["validation"][1]["source_row"])
    if ids_train & ids_val:
        raise ValueError("Train/Validation overlap")
    X_train, tracking_train = data["train"]
    y_train = tracking_train[TARGET].astype(int)
    if y_train.nunique() != 2:
        raise ValueError("Training labels require both classes")
    models = {
        "logistic_regression": LogisticRegression(
            class_weight="balanced", max_iter=3000, random_state=SEED),
        "isolation_forest": IsolationForest(
            n_estimators=200, contamination="auto", random_state=SEED, n_jobs=1),
    }
    models["logistic_regression"].fit(X_train, y_train)
    # Unsupervised: no target (y) passed to Isolation Forest.
    models["isolation_forest"].fit(X_train)
    input_files = [config_file, input_dir / "split_manifest.csv"] + [
        input_dir / f"{name}_{kind}.csv" for name in SPLITS for kind in ("features", "tracking")
    ]
    hashes = {str(p): _hash(p) for p in input_files}
    summary = {
        "preprocessing_source_sha256": config.get("input_sha256"),
        "input_sha256": hashes, "seed": SEED, "training_rows": len(X_train),
        "training_positives": int(y_train.sum()),
        "models": {
            "logistic_regression": {"class_weight": "balanced", "max_iter": 3000,
                                    "score": "P(Machine failure = 1)"},
            "isolation_forest": {"n_estimators": 200, "contamination": "auto",
                                 "fit_uses_labels": False, "score": "-decision_function (higher=more anomalous)"},
        },
        "scored_splits": list(SPLITS), "test_touched": False,
        "threshold_selected": False, "evaluation_metrics_calculated": False,
        "sklearn_version": sklearn.__version__,
        "limitations": "Random stratified split; time/equipment independence unverified; prior full-data EDA exposed test to exploratory inspection.",
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".models-", dir=output_dir.parent) as temp:
        staging = Path(temp) / "result"
        staging.mkdir()
        for name, (X, tracking) in data.items():
            prediction = tracking.copy()
            prediction["logistic_failure_probability"] = models["logistic_regression"].predict_proba(X)[:, 1]
            prediction["isolation_anomaly_score"] = -models["isolation_forest"].decision_function(X)
            prediction.to_csv(staging / f"{name}_scores.csv", index=False)
        for name, model in models.items():
            joblib.dump(model, staging / f"{name}.joblib")
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if any(_hash(p) != hashes[str(p)] for p in input_files):
            raise RuntimeError("Preprocessing inputs changed during training")
        staging.rename(output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("artifacts/preprocessing"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/modeling"))
    args = parser.parse_args()
    result = run(args.input, args.output)
    print(json.dumps({k: result[k] for k in (
        "training_rows", "training_positives", "scored_splits", "test_touched")}, indent=2))


if __name__ == "__main__":
    main()
