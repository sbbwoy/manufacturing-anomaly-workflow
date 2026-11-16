"""Offline tests for the preprocessing entry point and leakage boundaries."""
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from preprocess import (BASE_FEATURES, DERIVED, IDS, NUMERIC, PROCESS, TARGET,
                        ProcessFeatures, build_preprocessor, run, split_indices)
from scripts.validate_ai4i_labels import MODES, sha256


def sample():
    """Create independent synthetic records with enough positives for stratification."""
    count = 100
    data = pd.DataFrame({"UID": np.arange(1, count + 1),
                         "Product ID": [f"P{i}" for i in range(count)],
                         "Type": np.resize(["L", "M", "H"], count)})
    for offset, column in enumerate(PROCESS):
        data[column] = np.arange(count, dtype=float) + 10 + offset
    data[TARGET] = np.resize([0, 0, 0, 0, 1], count)
    for mode in MODES:
        data[mode] = 0
    return data


class PreprocessingTests(unittest.TestCase):
    def test_schema_and_missing_values(self):
        data = sample()
        invalid = [data.iloc[:0], data.drop(columns=[PROCESS[0]])]
        for column in BASE_FEATURES:
            case = data.copy()
            case[column] = case[column].astype(object)
            case.loc[0, column] = None
            invalid.append(case)
        for value in [np.inf, -np.inf, "invalid", True]:
            case = data.copy()
            case[PROCESS[0]] = value
            invalid.append(case)
        duplicate = pd.concat([data, data[[PROCESS[0]]]], axis=1)
        invalid.append(duplicate)
        for case in invalid:
            with self.subTest(columns=list(case.columns)):
                with self.assertRaises(ValueError):
                    build_preprocessor().fit(case)

    def test_unknown_type_at_fit_and_transform(self):
        data = sample()
        pipeline = build_preprocessor().fit(data)
        data.loc[0, "Type"] = "X"
        with self.assertRaisesRegex(ValueError, "Unknown Type"):
            pipeline.transform(data)
        with self.assertRaisesRegex(ValueError, "Unknown Type"):
            build_preprocessor().fit(data)

    def test_derived_values_and_whitelist(self):
        data = sample()
        values = ProcessFeatures().fit_transform(data)
        np.testing.assert_allclose(values[DERIVED[0]], 1)
        np.testing.assert_allclose(values[DERIVED[1]],
                                   data[PROCESS[3]] * data[PROCESS[2]] * 2 * np.pi / 60)
        np.testing.assert_allclose(values[DERIVED[2]],
                                   data[PROCESS[3]] * data[PROCESS[4]])
        self.assertEqual(list(values), BASE_FEATURES + DERIVED)
        pipeline = build_preprocessor().fit(data)
        contaminated = data.copy()
        for column in IDS + [TARGET] + list(MODES):
            contaminated[column] = "poison"
        contaminated["RNF_derived"] = 999
        np.testing.assert_array_equal(pipeline.transform(data),
                                      pipeline.transform(contaminated))
        # Prediction needs no tracking columns, label, or failure flags.
        np.testing.assert_array_equal(pipeline.transform(data),
                                      pipeline.transform(data[BASE_FEATURES]))

    def test_split_reproducibility_and_class_balance(self):
        data = sample()
        first = split_indices(data[TARGET])
        second = split_indices(data[TARGET])
        self.assertEqual([len(first[n]) for n in first], [60, 20, 20])
        all_indices = np.concatenate(list(first.values()))
        self.assertEqual(len(set(all_indices)), len(data))
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])
            self.assertAlmostEqual(data.iloc[first[name]][TARGET].mean(), 0.2)

    def test_fit_only_train_and_no_refit_on_transform(self):
        data = sample()
        splits = split_indices(data[TARGET])
        train = data.iloc[splits["train"]]
        pipeline = build_preprocessor().fit(train)
        scaler = pipeline.named_steps["columns"].named_transformers_["numeric"]
        expected = ProcessFeatures().transform(train)[NUMERIC]
        np.testing.assert_allclose(scaler.mean_, expected.mean())
        before_mean, before_scale = scaler.mean_.copy(), scaler.scale_.copy()
        held_out = data.iloc[splits["test"]].copy()
        held_out[PROCESS] = held_out[PROCESS] + 1_000_000
        pipeline.transform(held_out)
        np.testing.assert_array_equal(scaler.mean_, before_mean)
        np.testing.assert_array_equal(scaler.scale_, before_scale)
        self.assertEqual(scaler.n_samples_seen_, len(train))

    def test_cli_core_audit_alignment_and_serialization(self):
        data = sample()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            source = path / "input.csv"
            data.to_csv(source, index=False)
            before = sha256(source)
            output = path / "output"
            summary = run(source, output)
            self.assertEqual(sha256(source), before)
            # All 20 synthetic positives lack flags; retain them for audit.
            self.assertEqual(summary["label_disagreements"], 20)
            manifest = pd.read_csv(output / "split_manifest.csv")
            self.assertEqual(len(manifest), 100)
            self.assertEqual(manifest.label_disagreement.sum(), 20)
            self.assertEqual(manifest[TARGET].sum(), data[TARGET].sum())
            restored = joblib.load(output / "preprocessor.joblib")
            for name, indices in split_indices(data[TARGET]).items():
                tracking = pd.read_csv(output / f"{name}_tracking.csv")
                features = pd.read_csv(output / f"{name}_features.csv")
                np.testing.assert_array_equal(tracking["UID"], data.iloc[indices]["UID"])
                np.testing.assert_allclose(features,
                                           restored.transform(data.iloc[indices]))
                self.assertFalse(set(features) & set(IDS + [TARGET] + list(MODES)))
            with self.assertRaises(FileExistsError):
                run(source, output)

    def test_run_rejects_invalid_training_schema_before_output(self):
        data = sample()
        cases = [data.drop(columns=[TARGET])]
        for column, value in [("UID", None), ("UID", 1), (TARGET, 2)]:
            invalid = data.copy()
            invalid[column] = value
            cases.append(invalid)
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "invalid.csv"
            output = Path(temp) / "output"
            for invalid in cases:
                invalid.to_csv(source, index=False)
                with self.assertRaises(ValueError):
                    run(source, output)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
