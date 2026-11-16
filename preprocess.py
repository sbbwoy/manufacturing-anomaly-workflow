"""Reproducible AI4I preprocessing. Run python preprocess.py."""
import argparse
import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from eda import IDS, PROCESS, TARGET, load_data
from scripts.validate_ai4i_labels import audit, sha256

BASE_FEATURES = ["Type"] + PROCESS
DERIVED = ["Temperature difference [K]", "Mechanical power [W]",
           "Torque x tool wear [Nm min]"]
NUMERIC = PROCESS + DERIVED
SEED = 42


class ProcessFeatures(TransformerMixin, BaseEstimator):
    """Validate six input features and derive three row-local process quantities."""

    def fit(self, X, y=None):
        """Validate training input without learning any distribution parameters."""
        self.transform(X)
        self.n_features_in_ = len(BASE_FEATURES)
        self.feature_names_in_ = np.array(BASE_FEATURES, dtype=object)
        return self

    def transform(self, X):
        """Return nine whitelisted columns. Raise ValueError on invalid input."""
        if not isinstance(X, pd.DataFrame) or X.empty or X.columns.duplicated().any():
            raise ValueError("Expected a nonempty DataFrame with unique columns")
        missing = set(BASE_FEATURES) - set(X.columns)
        if missing:
            raise ValueError(f"Missing input columns: {sorted(missing)}")
        values = X[BASE_FEATURES].copy()
        if values.isna().any().any():
            raise ValueError("Missing process values or Type are not supported")
        if not values["Type"].isin(["L", "M", "H"]).all():
            raise ValueError("Unknown Type. Expected L, M or H")
        for column in PROCESS:
            if (not pd.api.types.is_numeric_dtype(values[column])
                    or pd.api.types.is_bool_dtype(values[column])):
                raise ValueError(f"Expected numeric input: {column}")
        if not np.isfinite(values[PROCESS].to_numpy(dtype=float)).all():
            raise ValueError("Process values must be finite")
        values[DERIVED[0]] = values[PROCESS[1]] - values[PROCESS[0]]
        values[DERIVED[1]] = values[PROCESS[3]] * values[PROCESS[2]] * (2 * np.pi / 60)
        values[DERIVED[2]] = values[PROCESS[3]] * values[PROCESS[4]]
        if not np.isfinite(values[NUMERIC].to_numpy(dtype=float)).all():
            raise ValueError("Derived process values must be finite")
        return values


def build_preprocessor():
    """Return a fresh pipeline. Only callers' training data may be used in fit."""
    columns = ColumnTransformer([
        ("type", OneHotEncoder(categories=[["L", "M", "H"]],
                               handle_unknown="error", sparse_output=False), ["Type"]),
        ("numeric", StandardScaler(), NUMERIC),
    ], remainder="drop")
    return Pipeline([("features", ProcessFeatures()), ("columns", columns)])


def split_indices(y):
    """Return disjoint 60/20/20 stratified row positions using seed 42 twice."""
    positions = np.arange(len(y))
    train, held_out = train_test_split(
        positions, test_size=0.4, stratify=y, random_state=SEED)
    validation, test = train_test_split(
        held_out, test_size=0.5, stratify=y.iloc[held_out], random_state=SEED)
    return {"train": train, "validation": validation, "test": test}


def run(input_path, output_path):
    """Write a new artifact directory atomically; refuse existing output paths."""
    input_path, output_path = Path(input_path), Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")
    protected = [input_path] + [p for p in Path("artifacts").rglob("*") if p.is_file()]
    before = {str(p): sha256(p) for p in protected}
    data = load_data(input_path)
    ProcessFeatures().transform(data)
    if data[IDS].isna().any().any() or data["UID"].duplicated().any():
        raise ValueError("Tracking identifiers must be present and UID must be unique")
    y = data[TARGET].copy()
    splits = split_indices(y)
    preprocessor = build_preprocessor()
    preprocessor.fit(data.iloc[splits["train"]][BASE_FEATURES])
    names = preprocessor.named_steps["columns"].get_feature_names_out()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".preprocess-", dir=output_path.parent) as temp:
        staging = Path(temp) / "result"
        staging.mkdir()
        # Reuse the established label auditor. Do not reimplement label OR logic.
        audit_summary = audit(input_path, staging / "audit")
        discrepancies = pd.read_csv(staging / "audit/label_discrepancies.csv")
        disagreement_rows = set((discrepancies["csv_row"] - 2).astype(int))
        memberships = []
        split_summary = {}
        for name, indices in splits.items():
            features = pd.DataFrame(
                preprocessor.transform(data.iloc[indices][BASE_FEATURES]),
                columns=names)
            features.to_csv(staging / f"{name}_features.csv", index=False)
            tracking = data.iloc[indices][IDS + [TARGET]].copy()
            tracking["source_row"] = indices
            tracking["csv_row"] = indices + 2
            tracking["label_disagreement"] = [int(i) in disagreement_rows for i in indices]
            tracking["split"] = name
            tracking.to_csv(staging / f"{name}_tracking.csv", index=False)
            memberships.append(tracking)
            split_summary[name] = {
                "rows": len(indices), "positive": int(y.iloc[indices].sum()),
                "disagreements": int(tracking.label_disagreement.sum())}
        pd.concat(memberships).sort_values("source_row").to_csv(
            staging / "split_manifest.csv", index=False)
        joblib.dump(preprocessor, staging / "preprocessor.joblib")
        summary = {
            "input_sha256": before[str(input_path)], "seed": SEED,
            "target": TARGET, "feature_names": list(names), "splits": split_summary,
            "label_disagreements": audit_summary["discrepancy_rows"],
            "versions": {"sklearn": sklearn.__version__, "pandas": pd.__version__,
                         "numpy": np.__version__}}
        (staging / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
        after = {p: sha256(Path(p)) for p in before}
        if before != after:
            raise RuntimeError("Protected data or existing artifacts changed")
        (staging / "preservation.json").write_text(
            json.dumps({"unchanged": True, "before": before, "after": after}, indent=2),
            encoding="utf-8")
        # Windows rename refuses a concurrently created destination.
        staging.rename(output_path)
    return summary


def main():
    """CLI entry point for creating a new preprocessing run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/ai4i2020.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/preprocessing"))
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output), indent=2))


if __name__ == "__main__":
    # Persist the transformer with its importable module name for later reloads.
    from preprocess import main as execute
    execute()
