"""Quality gates must not fabricate certainty from empty denominators."""

import math
import unittest

from validation_patterns.quality import Counts, QualityGate, assess_quality


class QualityTests(unittest.TestCase):
    def test_balanced_synthetic_sample_passes(self):
        counts = Counts(19, 1, 0, 20)
        report = assess_quality(counts, min_recall=0.9)
        self.assertEqual(report.gate, QualityGate.PASS)
        self.assertAlmostEqual(report.recall, 0.95)
        self.assertEqual(report.false_positive_rate, 0.0)
        self.assertEqual(report.precision, 1.0)

    def test_missed_defects_trigger_failure(self):
        report = assess_quality(Counts(1, 9, 0, 20))
        self.assertEqual(report.gate, QualityGate.FAIL)
        self.assertAlmostEqual(report.recall, 0.1)

    def test_excessive_false_alarms_trigger_failure(self):
        report = assess_quality(Counts(20, 0, 3, 17))
        self.assertEqual(report.gate, QualityGate.FAIL)
        self.assertAlmostEqual(report.false_positive_rate, 0.15)

    def test_only_clean_inputs_cannot_pass(self):
        report = assess_quality(Counts(0, 0, 0, 100))
        self.assertEqual(report.gate, QualityGate.INCONCLUSIVE)
        self.assertIsNone(report.recall)
        self.assertEqual(report.false_positive_rate, 0)

    def test_only_bad_inputs_cannot_pass(self):
        report = assess_quality(Counts(30, 0, 0, 0))
        self.assertEqual(report.gate, QualityGate.INCONCLUSIVE)
        self.assertIsNone(report.false_positive_rate)

    def test_no_inputs_cannot_pass(self):
        report = assess_quality(Counts(0, 0, 0, 0))
        self.assertEqual(report.gate, QualityGate.INCONCLUSIVE)
        self.assertIsNone(report.evidence_coverage)
        self.assertIsNone(report.recall)

    def test_unknown_ground_truth_preserved(self):
        report = assess_quality(Counts(20, 0, 0, 20, unknown=10))
        self.assertEqual(report.gate, QualityGate.INCONCLUSIVE)
        self.assertAlmostEqual(report.evidence_coverage, 0.8)
        self.assertEqual(report.to_dict()["unknown_cases"], 10)

    def test_known_failure_is_not_hidden_by_unknowns(self):
        report = assess_quality(Counts(1, 19, 0, 20, unknown=5))
        self.assertEqual(report.gate, QualityGate.FAIL)

    def test_undersized_sample_is_inconclusive(self):
        report = assess_quality(Counts(2, 0, 0, 2), min_known_cases=10)
        self.assertEqual(report.gate, QualityGate.INCONCLUSIVE)

    def test_negative_counter_is_rejected(self):
        with self.assertRaises(ValueError):
            Counts(0, -1, 0, 0)

    def test_boolean_counter_is_rejected(self):
        with self.assertRaises(ValueError):
            Counts(True, 0, 0, 0)

    def test_fractional_counter_is_rejected(self):
        with self.assertRaises(ValueError):
            Counts(1.0, 0, 0, 0)

    def test_minimum_case_requirement_is_validated(self):
        with self.assertRaises(ValueError):
            assess_quality(Counts(1, 0, 0, 1), min_known_cases=0)

    def test_invalid_threshold_is_rejected(self):
        with self.assertRaises(ValueError):
            assess_quality(Counts(1, 0, 0, 1), min_recall=1.5)

    def test_nan_threshold_is_rejected(self):
        with self.assertRaises(ValueError):
            assess_quality(Counts(1, 0, 0, 1), min_recall=math.nan)

    def test_metric_snapshot_is_plain_data(self):
        snapshot = assess_quality(Counts(8, 2, 2, 8)).to_dict()
        self.assertEqual(snapshot["known_cases"], 20)
        self.assertEqual(snapshot["gate"], "FAIL")


if __name__ == "__main__":
    unittest.main()