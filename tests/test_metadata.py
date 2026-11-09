"""Small synthetic fixtures; no ECG downloads or third-party packages."""

import csv
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ecg_lbbb.inspect_metadata import inspect, parse_codes


class MetadataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        with (self.root / "scp_statements.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["", "description", "diagnostic"])
            writer.writerows([
                ["CLBBB", "complete left bundle branch block", 1],
                ["ILBBB", "incomplete left bundle branch block", 1],
                ["NORM", "normal ECG", 1],
            ])

    def database(self, records):
        with (self.root / "ptbxl_database.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["ecg_id", "patient_id", "strat_fold", "scp_codes", "filename_lr", "filename_hr"])
            for index, (patient, fold, codes) in enumerate(records, 1):
                writer.writerow([index, patient, fold, repr(codes), "unused_lr", "unused_hr"])

    def test_counts_conflicts_and_zero_likelihood(self):
        self.database([(1, 1, {"CLBBB": 100, "ILBBB": 0}),
                       (2, 9, {"NORM": 100}), (3, 10, {"CLBBB": 100, "NORM": 50})])
        report = inspect(self.root)
        self.assertEqual(report["records"], 3)
        self.assertEqual(report["patients_in_multiple_folds"], 0)
        self.assertEqual(report["conflicts_by_code_presence"]["CLBBB_with_ILBBB"], 1)
        self.assertEqual(report["conflicts_by_code_presence"]["NORM_with_other_diagnostic"], 1)

    def test_leakage_rejected_even_with_numeric_id_formatting(self):
        self.database([("1", 1, {"NORM": 100}), ("1.0", 10, {"CLBBB": 100})])
        with self.assertRaisesRegex(ValueError, "Patient leakage"):
            inspect(self.root)

    def test_unknown_code_rejected(self):
        self.database([(1, 1, {"LBBB": 100})])
        with self.assertRaisesRegex(ValueError, "Unknown codes"):
            inspect(self.root)

    def test_invalid_likelihoods_rejected(self):
        for value in ["[]", "{}", "{'NORM': -1}", "{'NORM': 101}", "{'NORM': True}"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_codes(value)

    def test_missing_files(self):
        with self.assertRaises(FileNotFoundError):
            inspect(self.root)


if __name__ == "__main__":
    unittest.main()
