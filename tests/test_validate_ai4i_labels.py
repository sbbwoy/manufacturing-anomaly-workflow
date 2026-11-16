"""Standard-library tests: python -m unittest discover -s tests -v"""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from validate_ai4i_labels import audit, sha256  # noqa: E402


class LabelAuditTests(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.TemporaryDirectory()
        self.addCleanup(self.work.cleanup)
        self.base = Path(self.work.name)
        self.csv_path = self.base / "ai4i.csv"

    def write_rows(self, rows):
        fieldnames = ["UDI", "Product ID", "Type", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
        with self.csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def row(self, id_, target, flags=()):
        return {"UDI": str(id_), "Product ID": f"M{id_}", "Type": "M", "Machine failure": str(target),
                **{m: str(int(m in flags)) for m in ("TWF", "HDF", "PWF", "OSF", "RNF")}}

    def test_detects_both_discrepancy_types_and_preserves_input(self):
        self.write_rows([self.row(1, 0), self.row(2, 1, ("HDF",)),
                         self.row(3, 0, ("TWF",)), self.row(4, 1)])
        before = sha256(self.csv_path)
        summary = audit(self.csv_path, self.base / "reports")
        self.assertEqual(2, summary["discrepancy_rows"])
        self.assertEqual(1, summary["discrepancy_counts"]["mode_flag_but_normal"])
        self.assertEqual(1, summary["discrepancy_counts"]["failure_without_mode_flag"])
        self.assertEqual(before, sha256(self.csv_path))
        with (self.base / "reports" / "label_discrepancies.csv").open(newline="") as handle:
            self.assertEqual({"3", "4"}, {r["UDI_or_UID"] for r in csv.DictReader(handle)})
        self.assertEqual(4, json.loads((self.base / "reports" / "label_audit.json").read_text())["total_rows"])

    def test_multiple_failure_flags_count_once(self):
        self.write_rows([self.row(1, 1, ("TWF", "HDF"))])
        summary = audit(self.csv_path, self.base / "reports")
        self.assertEqual(1, summary["rows_with_multiple_mode_flags"])
        self.assertEqual(0, summary["discrepancy_rows"])

    def test_rejects_invalid_target(self):
        self.write_rows([self.row(1, 2)])
        with self.assertRaisesRegex(ValueError, "non-binary"):
            audit(self.csv_path, self.base / "reports")

    def test_rejects_missing_columns(self):
        self.csv_path.write_text("Machine failure,TWF\n0,0\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Missing required columns"):
            audit(self.csv_path, self.base / "reports")


if __name__ == "__main__":
    unittest.main()
