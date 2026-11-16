"""Synthetic holdout tests. Never open real Test inputs."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from evaluate_holdout import PINNED_PATHS, THRESHOLD, run
from scripts.validate_ai4i_labels import sha256
from train_models import FEATURES, TARGET


def make_fixture(root):
    """Create a small synthetic trained model and independent frozen holdout."""
    model_dir = root / "artifacts/modeling"
    preproc = root / "artifacts/preprocessing"
    eval_dir = root / "artifacts/evaluation"
    for directory in [model_dir, preproc, eval_dir, root / "config"]:
        directory.mkdir(parents=True)
    rng = np.random.default_rng(42)
    X_train = pd.DataFrame(rng.normal(size=(80, len(FEATURES))), columns=FEATURES)
    model = LogisticRegression(random_state=42).fit(X_train, np.arange(80) % 2)
    joblib.dump(model, model_dir / "logistic_regression.joblib")
    (model_dir / "summary.json").write_text("{}")
    manifest = []
    for name, count, offset in [("train", 80, 0), ("validation", 24, 80), ("test", 24, 104)]:
        tracking = pd.DataFrame({
            "UID": np.arange(offset + 1, offset + count + 1),
            "Product ID": [f"L{i}" for i in range(offset, offset + count)],
            TARGET: np.arange(count) % 2,
            "source_row": np.arange(offset, offset + count),
            "csv_row": np.arange(offset + 2, offset + count + 2),
            "label_disagreement": [False, True] + [False] * (count - 2),
            "split": name})
        manifest.append(tracking)
        if name == "test":
            tracking.to_csv(preproc / "test_tracking.csv", index=False)
            pd.DataFrame(rng.normal(size=(count, len(FEATURES))), columns=FEATURES).to_csv(
                preproc / "test_features.csv", index=False)
    pd.concat(manifest).to_csv(preproc / "split_manifest.csv", index=False)
    (preproc / "summary.json").write_text(json.dumps(
        {"target": TARGET, "feature_names": FEATURES, "input_sha256": "synthetic"}))
    pd.DataFrame({"model": ["logistic_regression", "isolation_forest"],
                  "ap": [0.8, 0.2], "threshold": [THRESHOLD, 0.1]}).to_csv(
                      eval_dir / "metrics.csv", index=False)
    config = {
        "model": "logistic_regression", "threshold": THRESHOLD,
        "prediction_rule": "score >= threshold",
        "selection_reason": "Highest Validation AP; threshold maximizes Validation F2",
        "expected_rows": 24,
        "pinned_sha256": {p: sha256(root / p) for p in PINNED_PATHS}}
    config_path = root / "config/final_holdout.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


class HoldoutTests(unittest.TestCase):
    def test_fixed_inference_once_preserves_inputs_and_errors(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = make_fixture(root)
            protected = {str(p): sha256(p) for p in root.rglob("*") if p.is_file()}
            original_predict = LogisticRegression.predict_proba
            calls = []
            def predict(model, features):
                calls.append(len(features))
                return original_predict(model, features)
            with patch.object(LogisticRegression, "fit", side_effect=AssertionError("No fit")), \
                    patch.object(LogisticRegression, "predict_proba", new=predict), \
                    patch("evaluate_validation.choose_threshold", side_effect=AssertionError("No tuning")):
                summary = run(config)
            self.assertEqual(calls, [24])
            self.assertEqual(summary["predict_proba_calls"], 1)
            self.assertEqual(summary["metrics"]["threshold"], THRESHOLD)
            self.assertEqual(summary["metrics"]["rows"], 24)
            self.assertEqual(summary["label_disagreements"], 1)
            for path, expected in protected.items():
                self.assertEqual(sha256(Path(path)), expected)
            predictions = pd.read_csv(root / "artifacts/final_holdout/test_predictions.csv")
            errors = pd.read_csv(root / "artifacts/final_holdout/test_errors.csv")
            expected_errors = predictions.loc[predictions.predicted_failure.ne(predictions[TARGET])]
            self.assertEqual(list(errors.UID), list(expected_errors.UID))
            self.assertEqual(len(errors), summary["metrics"]["fp"] + summary["metrics"]["fn"])
            self.assertTrue(errors.loc[errors.error_type.eq("FN"), TARGET].eq(1).all())
            matrix = pd.read_csv(root / "artifacts/final_holdout/confusion_matrix.csv", index_col=0)
            np.testing.assert_array_equal(matrix.to_numpy().ravel(),
                [summary["metrics"][n] for n in ["tn", "fp", "fn", "tp"]])
            # Repeat is blocked before model or Test reads.
            with patch("evaluate_holdout.joblib.load", side_effect=AssertionError("No repeat")):
                with self.assertRaises(FileExistsError):
                    run(config)
            ledger = root / "artifacts/final_holdout_execution.json"
            self.assertEqual(json.loads(ledger.read_text())["status"], "completed")

    def test_policy_and_hash_tamper_rejected_before_test_read(self):
        for tamper in ["threshold", "model_hash"]:
            with self.subTest(tamper=tamper), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                config_path = make_fixture(root)
                config = json.loads(config_path.read_text())
                if tamper == "threshold":
                    config["threshold"] = 0.5
                else:
                    config["pinned_sha256"][PINNED_PATHS[0]] = "bad"
                config_path.write_text(json.dumps(config))
                with patch("evaluate_holdout.load_split", side_effect=AssertionError("No Test")):
                    with self.assertRaises(ValueError):
                        run(config_path)
                self.assertFalse((root / "artifacts/final_holdout_execution.json").exists())

    def test_failed_claim_persists_and_blocks_retry(self):
        for failure in ["tracking", "schema", "missing", "manifest_overlap", "nonfinite"]:
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                config = make_fixture(root)
                preproc = root / "artifacts/preprocessing"
                if failure == "tracking":
                    tracking = pd.read_csv(preproc / "test_tracking.csv")
                    tracking.loc[0, "UID"] = 999
                    tracking.to_csv(preproc / "test_tracking.csv", index=False)
                elif failure == "manifest_overlap":
                    manifest = pd.read_csv(preproc / "split_manifest.csv")
                    manifest.loc[len(manifest)-1, "UID"] = manifest.loc[0, "UID"]
                    manifest.to_csv(preproc / "split_manifest.csv", index=False)
                else:
                    features = pd.read_csv(preproc / "test_features.csv")
                    if failure == "schema":
                        features = features.drop(columns=[FEATURES[0]])
                    else:
                        features.iloc[0, 4] = np.nan if failure == "missing" else np.inf
                    features.to_csv(preproc / "test_features.csv", index=False)
                with self.assertRaises(ValueError):
                    run(config)
                ledger = root / "artifacts/final_holdout_execution.json"
                self.assertEqual(json.loads(ledger.read_text())["status"], "failed")
                self.assertFalse((root / "artifacts/final_holdout").exists())
                with self.assertRaises(FileExistsError):
                    run(config)

    def test_prior_claim_blocks_even_without_result(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = make_fixture(root)
            ledger = root / "artifacts/final_holdout_execution.json"
            ledger.write_text('{"status":"claimed"}')
            with patch("evaluate_holdout.sha256", side_effect=AssertionError("No input read")):
                with self.assertRaises(FileExistsError):
                    run(config)


if __name__ == "__main__":
    unittest.main()
