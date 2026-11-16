"""Synthetic-only tests for validation evaluation and holdout access."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, fbeta_score, roc_auc_score

from evaluate_validation import (MODELS, TARGET, choose_threshold, metrics, run,
                                 validate_scores)
from scripts.validate_ai4i_labels import sha256


def fixture():
    return pd.DataFrame({
        "UID": [101, 102, 103, 104, 105, 106],
        "Product ID": ["L1", "M2", "H3", "L4", "M5", "H6"],
        TARGET: [0, 1, 0, 1, 0, 1],
        "source_row": range(6), "csv_row": range(2, 8),
        "label_disagreement": [False, True, False, False, True, False],
        "split": ["validation"] * 6,
        "logistic_failure_probability": [0.1, 0.8, 0.5, 0.7, 0.3, 0.4],
        "isolation_anomaly_score": [-0.2, 0.1, 0.0, 0.2, -0.1, -0.3],
    })


class EvaluationTests(unittest.TestCase):
    def test_f2_threshold_matches_exhaustive_search_and_ties(self):
        data = fixture()
        for column in MODELS.values():
            scores, y = data[column], data[TARGET]
            candidates = [(fbeta_score(y, scores >= t, beta=2), t)
                          for t in np.unique(scores)]
            self.assertEqual(choose_threshold(y, scores), max(candidates)[1])
        # Exact tie: F2=5/9 for (TP=1,FP=0,FN=1) and (TP=2,FP=8,FN=0).
        y = np.array([1] + [0] * 8 + [1])
        scores = np.array([1.] + [0.] * 9)
        self.assertEqual(choose_threshold(y, scores), 1.)
        self.assertEqual(choose_threshold([0, 1], [0.2, 0.2]), 0.2)

    def test_metrics_known_values_and_single_class(self):
        result = metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.8, 0.3, 0.9]), 0.5)
        self.assertEqual([result[n] for n in ["tn", "fp", "fn", "tp"]], [1, 1, 1, 1])
        for name in ["precision", "recall", "f1", "f2"]:
            self.assertAlmostEqual(result[name], 0.5)
        single = metrics([0, 0], [0.1, 0.2], 0.5)
        self.assertIsNone(single["ap"])
        self.assertIsNone(single["roc_auc"])
        self.assertEqual(single["f2"], 0)

    def test_invalid_scores_schema_and_tracking(self):
        data = fixture()
        cases = [data.iloc[:0], data.drop(columns=[TARGET])]
        for column, value in [
            ("split", "test"), ("UID", 1), ("csv_row", 2), (TARGET, 2),
            ("label_disagreement", "False"), ("isolation_anomaly_score", np.inf),
            ("logistic_failure_probability", 1.5),
            ("isolation_anomaly_score", "bad"), ("Product ID", None),
        ]:
            case = data.copy()
            case[column] = value
            cases.append(case)
        case = data.copy()
        case[TARGET] = 0
        cases.append(case)
        for case in cases:
            with self.subTest(columns=list(case)):
                with self.assertRaises(ValueError):
                    validate_scores(case)

    def test_run_preserves_input_aligns_errors_and_fixes_sensitivity_threshold(self):
        data = fixture()
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            source = folder / "validation_scores.csv"
            data.to_csv(source, index=False)
            before = sha256(source)
            real_read = pd.read_csv
            reads = []
            def guarded_read(path, *args, **kwargs):
                self.assertEqual(Path(path).resolve(), source.resolve())
                reads.append(str(path))
                return real_read(path, *args, **kwargs)
            with patch("evaluate_validation.pd.read_csv", side_effect=guarded_read):
                summary = run(source, folder / "out")
            self.assertEqual(len(reads), 1)
            self.assertFalse(summary["test_read"])
            self.assertEqual(sha256(source), before)
            sensitivity = pd.read_csv(folder / "out/sensitivity.csv")
            for model, column in MODELS.items():
                result = next(r for r in summary["metrics"] if r["model"] == model)
                self.assertAlmostEqual(result["ap"], average_precision_score(data[TARGET], data[column]))
                self.assertAlmostEqual(result["roc_auc"], roc_auc_score(data[TARGET], data[column]))
                rows = sensitivity.loc[sensitivity.model.eq(model)]
                self.assertTrue(rows.threshold.eq(result["threshold"]).all())
                alternate = np.where(data.label_disagreement, 1 - data[TARGET], data[TARGET])
                alternative_row = rows.loc[rows.scenario.eq("flag_or_alternative")].iloc[0]
                self.assertEqual(alternative_row.positives, alternate.sum())
                expected_f2 = fbeta_score(alternate, data[column] >= result["threshold"], beta=2)
                self.assertAlmostEqual(alternative_row.f2, expected_f2)
                errors = pd.read_csv(folder / f"out/{model}_errors.csv")
                expected = data.loc[(data[column] >= result["threshold"]) != data[TARGET]]
                self.assertEqual(list(errors.UID), list(expected.UID))
                self.assertEqual(len(errors), result["fp"] + result["fn"])
            with self.assertRaises(FileExistsError):
                run(source, folder / "out")
            with self.assertRaises(ValueError):
                run(folder / "test_scores.csv", folder / "bad")
            # All inconsistent observations may be excluded; report undefined ranking.
            data["label_disagreement"] = True
            data.to_csv(source, index=False)
            run(source, folder / "all_disagree")


if __name__ == "__main__":
    unittest.main()
