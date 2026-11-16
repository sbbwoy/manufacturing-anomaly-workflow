"""Offline tests for model training, traceability and holdout isolation."""
import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from train_models import FEATURES, TARGET, run


def make_inputs(root: Path, bad=False):
    folder = root / "preprocessing"
    folder.mkdir()
    rng = np.random.default_rng(123)
    manifest = []
    for split, n, offset in (("train", 80, 0), ("validation", 24, 80), ("test", 18, 104)):
        X = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
        X[FEATURES[0:3]] = np.eye(3)[np.arange(n) % 3]
        if bad and split == "train":
            X.iloc[0, 3] = np.nan
        y = (np.arange(n) % 4 == 0).astype(int)
        t = pd.DataFrame({"UID": np.arange(offset + 1, offset + n + 1),
                          "Product ID": [f"ID{i}" for i in range(offset, offset + n)],
                          TARGET: y, "source_row": np.arange(offset, offset + n),
                          "csv_row": np.arange(offset + 2, offset + n + 2),
                          "label_disagreement": [False] * n, "split": [split] * n})
        manifest.append(t)
        if split != "test":
            X.to_csv(folder / f"{split}_features.csv", index=False)
            t.to_csv(folder / f"{split}_tracking.csv", index=False)
    pd.concat(manifest).to_csv(folder / "split_manifest.csv", index=False)
    (folder / "summary.json").write_text(json.dumps({
        "target": TARGET, "feature_names": FEATURES, "input_sha256": "fake-source-hash"}))
    return folder


class ModelTrainingTests(unittest.TestCase):
    def test_scores_and_reproducibility_no_test_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            data = make_inputs(base)
            first = run(data, base / "model1")
            second = run(data, base / "model2")
            self.assertEqual(first["training_rows"], 80)
            self.assertFalse(first["test_touched"])
            self.assertFalse((base / "model1" / "test_scores.csv").exists())
            for split, n in (("train", 80), ("validation", 24)):
                a = pd.read_csv(base / "model1" / f"{split}_scores.csv")
                b = pd.read_csv(base / "model2" / f"{split}_scores.csv")
                self.assertEqual(len(a), n)
                self.assertEqual(list(a["source_row"]), list(pd.read_csv(data / f"{split}_tracking.csv")["source_row"]))
                np.testing.assert_allclose(a["logistic_failure_probability"], b["logistic_failure_probability"])
                np.testing.assert_allclose(a["isolation_anomaly_score"], b["isolation_anomaly_score"])
                self.assertTrue(a["logistic_failure_probability"].between(0, 1).all())
            self.assertTrue(hasattr(joblib.load(base / "model1" / "logistic_regression.joblib"), "classes_"))
            with self.assertRaises(FileExistsError):
                run(data, base / "model1")

    def test_rejects_invalid_data_and_manifest_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            data = make_inputs(base, bad=True)
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                run(data, base / "bad1")
            self.assertFalse((base / "bad1").exists())
            X = pd.read_csv(data / "train_features.csv").fillna(0)
            X.to_csv(data / "train_features.csv", index=False)
            tracking = pd.read_csv(data / "train_tracking.csv")
            tracking.loc[0, "UID"] = 99999
            tracking.to_csv(data / "train_tracking.csv", index=False)
            with self.assertRaisesRegex(ValueError, "manifest"):
                run(data, base / "bad2")

    def test_isolation_forest_fits_without_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            data = make_inputs(base)
            run(data, base / "v1")
            for split in ("train", "validation"):
                t = pd.read_csv(data / f"{split}_tracking.csv")
                t[TARGET] = 1 - t[TARGET]
                t.to_csv(data / f"{split}_tracking.csv", index=False)
            manifest = pd.read_csv(data / "split_manifest.csv")
            manifest[TARGET] = 1 - manifest[TARGET]
            manifest.to_csv(data / "split_manifest.csv", index=False)
            run(data, base / "v2")
            for split in ("train", "validation"):
                a = pd.read_csv(base / "v1" / f"{split}_scores.csv")
                b = pd.read_csv(base / "v2" / f"{split}_scores.csv")
                np.testing.assert_allclose(a["isolation_anomaly_score"], b["isolation_anomaly_score"])


if __name__ == "__main__":
    unittest.main()
