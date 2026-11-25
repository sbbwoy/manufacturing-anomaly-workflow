"""Synthetic tests for the unsupervised comparison and Test isolation."""
import hashlib
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

import compare_unsupervised as comparison
from train_models import FEATURES, TARGET, TRACK_COLS


def make_inputs(root: Path) -> Path:
    """Write small frozen Train/Validation inputs and poison Test sentinels."""
    folder = root / "preprocessing"
    folder.mkdir()
    rng = np.random.default_rng(123)
    for split, rows, offset in (("train", 100, 0), ("validation", 32, 100), ("test", 12, 132)):
        features = pd.DataFrame(rng.normal(size=(rows, len(FEATURES))), columns=FEATURES)
        features[FEATURES[:3]] = np.eye(3)[np.arange(rows) % 3]
        labels = np.arange(rows) % 2
        tracking = pd.DataFrame({
            "UID": np.arange(offset + 1, offset + rows + 1),
            "Product ID": [f"ID{i}" for i in range(offset, offset + rows)],
            TARGET: labels,
            "source_row": np.arange(offset, offset + rows),
            "csv_row": np.arange(offset + 2, offset + rows + 2),
            "label_disagreement": np.resize([False, True], rows),
            "split": [split] * rows,
        })[TRACK_COLS]
        features.to_csv(folder / f"{split}_features.csv", index=False)
        tracking.to_csv(folder / f"{split}_tracking.csv", index=False)
    return folder


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class UnsupervisedComparisonTests(unittest.TestCase):
    def test_models_fit_without_labels_scores_reproducibly_and_skip_test(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = make_inputs(root)
            test_hashes = {name: sha256(source / name) for name in
                           ("test_features.csv", "test_tracking.csv")}
            fit_calls = []
            score_complete = False
            original_fit_and_score = comparison._fit_and_score
            original_read_labels = comparison._load_validation_labels
            original_read_csv = pd.read_csv

            def guarded_read_csv(path, *args, **kwargs):
                self.assertNotIn(Path(path).name, {"test_features.csv", "test_tracking.csv"})
                if (Path(path).name == "validation_tracking.csv"
                        and kwargs.get("usecols") == [TARGET]):
                    self.assertTrue(score_complete, "Validation labels were loaded before scoring")
                return original_read_csv(path, *args, **kwargs)

            def fit_and_mark(*args, **kwargs):
                nonlocal score_complete
                score_complete = False
                result = original_fit_and_score(*args, **kwargs)
                score_complete = True
                return result

            def read_labels_after_scoring(*args, **kwargs):
                self.assertTrue(score_complete)
                return original_read_labels(*args, **kwargs)

            estimator_types = (IsolationForest, LocalOutlierFactor, OneClassSVM, PCA)
            with ExitStack() as stack:
                for estimator_type in estimator_types:
                    original_fit = estimator_type.fit

                    def fit_without_y(estimator, X, y=None, *args, _fit=original_fit, **kwargs):
                        self.assertIsNone(y)
                        fit_calls.append(type(estimator).__name__)
                        return _fit(estimator, X, y, *args, **kwargs)

                    stack.enter_context(patch.object(estimator_type, "fit", new=fit_without_y))
                stack.enter_context(patch("compare_unsupervised.pd.read_csv",
                                          side_effect=guarded_read_csv))
                stack.enter_context(patch.object(comparison, "_fit_and_score",
                                                 side_effect=fit_and_mark))
                stack.enter_context(patch.object(comparison, "_load_validation_labels",
                                                 side_effect=read_labels_after_scoring))
                first = comparison.run(source, root / "out1")
                second = comparison.run(source, root / "out2")

            self.assertEqual(len(fit_calls), 8)
            self.assertEqual(set(first["input_sha256"]), {
                "train_features.csv", "train_tracking.csv",
                "validation_features.csv", "validation_tracking.csv"})
            self.assertFalse(first["fit_uses_labels"])
            self.assertTrue(first["validation_labels_loaded_after_scoring"])
            self.assertFalse(first["test_touched"])
            self.assertTrue(first["models"]["lof"]["novelty"])
            for model_name, score_column in comparison.SCORE_COLUMNS.items():
                first_scores = pd.read_csv(root / "out1/validation_scores.csv")[score_column]
                second_scores = pd.read_csv(root / "out2/validation_scores.csv")[score_column]
                np.testing.assert_array_equal(first_scores, second_scores)
            scored = pd.read_csv(root / "out1/validation_scores.csv")
            features = pd.read_csv(source / "validation_features.csv")
            models = {name: joblib.load(root / f"out1/{name}.joblib")
                      for name in comparison.SCORE_COLUMNS}
            self.assertTrue(models["lof"].novelty)
            np.testing.assert_allclose(scored[comparison.SCORE_COLUMNS["isolation_forest"]],
                                       -models["isolation_forest"].decision_function(
                                           features.to_numpy()))
            np.testing.assert_allclose(scored[comparison.SCORE_COLUMNS["lof"]],
                                       -models["lof"].score_samples(features.to_numpy()))
            np.testing.assert_allclose(scored[comparison.SCORE_COLUMNS["one_class_svm"]],
                                       -models["one_class_svm"].decision_function(
                                           features.to_numpy()))
            matrix = features.to_numpy()
            reconstructed = models["pca"].inverse_transform(models["pca"].transform(matrix))
            expected_error = np.mean(np.square(matrix - reconstructed), axis=1)
            np.testing.assert_allclose(scored[comparison.SCORE_COLUMNS["pca"]], expected_error)
            metrics = pd.read_csv(root / "out1/metrics.csv").set_index("model")
            second_metrics = pd.read_csv(root / "out2/metrics.csv").set_index("model")
            for column in ("ap", "roc_auc", "threshold", "precision", "recall", "fpr"):
                np.testing.assert_array_equal(metrics[column], second_metrics[column])
            for filename in ("isolation_forest.joblib", "lof.joblib", "one_class_svm.joblib",
                             "pca.joblib", "validation_scores.csv", "metrics.csv",
                             "comparison.csv", "pr_curves.csv", "precision_recall_curves.png",
                             "summary.json", "report.md"):
                self.assertTrue((root / f"out1/{filename}").is_file())
            self.assertEqual(test_hashes, {name: sha256(source / name) for name in test_hashes})

    def test_threshold_is_label_independent_95th_percentile(self):
        scores = np.arange(20, dtype=float)
        positive_first = np.resize([1, 0], 20)
        positive_last = 1 - positive_first
        first, _ = comparison._evaluate_model(positive_first, scores)
        second, _ = comparison._evaluate_model(positive_last, scores)
        self.assertEqual(first["threshold"], np.quantile(scores, 0.95))
        self.assertEqual(first["threshold"], second["threshold"])
        self.assertEqual(first["threshold_quantile"], 0.95)

    def test_rejects_invalid_feature_and_tracking_inputs_before_output(self):
        for invalid in ("empty", "missing_feature", "missing_value", "wrong_split"):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source = make_inputs(root)
                features_path = source / "train_features.csv"
                tracking_path = source / "train_tracking.csv"
                if invalid == "empty":
                    pd.DataFrame(columns=FEATURES).to_csv(features_path, index=False)
                elif invalid == "missing_feature":
                    pd.read_csv(features_path).drop(columns=[FEATURES[0]]).to_csv(
                        features_path, index=False)
                elif invalid == "missing_value":
                    features = pd.read_csv(features_path)
                    features.loc[0, FEATURES[3]] = np.nan
                    features.to_csv(features_path, index=False)
                else:
                    tracking = pd.read_csv(tracking_path)
                    tracking.loc[0, "split"] = "validation"
                    tracking.to_csv(tracking_path, index=False)
                with self.assertRaises(ValueError):
                    comparison.run(source, root / "bad_output")
                self.assertFalse((root / "bad_output").exists())


if __name__ == "__main__":
    unittest.main()
