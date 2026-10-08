"""Direction and uncertainty are more important than an impressive pass count."""

import unittest

from validation_patterns.differential import (
    Change,
    Gate,
    State,
    compare_cases,
)


class DifferentialTests(unittest.TestCase):
    def test_preserved_refusal_is_pass(self):
        report = compare_cases({"one": "REFUSE"}, {"one": "REFUSE"})
        self.assertEqual(report.gate, Gate.PASS)
        self.assertEqual(report.count(Change.BOTH_REFUSE), 1)

    def test_preserved_acceptance_is_pass(self):
        report = compare_cases({"one": State.ACCEPT}, {"one": State.ACCEPT})
        self.assertEqual(report.gate, Gate.PASS)

    def test_lost_refusal_fails(self):
        report = compare_cases({"case": "REFUSE"}, {"case": "ACCEPT"})
        self.assertEqual(report.gate, Gate.FAIL)
        self.assertEqual(report.count(Change.MISSED_REFUSAL), 1)

    def test_new_refusal_needs_independent_review(self):
        report = compare_cases({"case": "ACCEPT"}, {"case": "REFUSE"})
        self.assertEqual(report.gate, Gate.REVIEW)
        self.assertEqual(report.count(Change.NEW_REFUSAL), 1)

    def test_no_cases_is_inconclusive(self):
        self.assertEqual(compare_cases({}, {}).gate, Gate.INCONCLUSIVE)

    def test_missing_new_case_is_inconclusive(self):
        report = compare_cases({"case": "REFUSE"}, {})
        self.assertEqual(report.gate, Gate.INCONCLUSIVE)

    def test_missing_reference_case_is_inconclusive(self):
        report = compare_cases({}, {"case": "ACCEPT"})
        self.assertEqual(report.gate, Gate.INCONCLUSIVE)

    def test_invalid_state_is_inconclusive(self):
        report = compare_cases({"case": "BANANA"}, {"case": "ACCEPT"})
        self.assertEqual(report.gate, Gate.INCONCLUSIVE)

    def test_explicit_unknown_is_inconclusive(self):
        report = compare_cases({"case": "UNKNOWN"}, {"case": "ACCEPT"})
        self.assertEqual(report.gate, Gate.INCONCLUSIVE)

    def test_internal_error_is_not_counted_as_pass(self):
        report = compare_cases({"case": "ERROR"}, {"case": "ACCEPT"})
        self.assertEqual(report.gate, Gate.INCONCLUSIVE)

    def test_real_regression_outweighs_inconclusive_data(self):
        report = compare_cases(
            {"broken": "REFUSE", "unknown": "UNKNOWN"},
            {"broken": "ACCEPT", "unknown": "ACCEPT"},
        )
        self.assertEqual(report.gate, Gate.FAIL)

    def test_unknown_outweighs_new_refusal_review(self):
        report = compare_cases(
            {"new": "ACCEPT", "other": "ACCEPT"},
            {"new": "REFUSE", "other": "UNKNOWN"},
        )
        self.assertEqual(report.gate, Gate.INCONCLUSIVE)

    def test_deterministic_case_sorting(self):
        report = compare_cases(
            {"z": "ACCEPT", "a": "REFUSE"},
            {"z": "ACCEPT", "a": "REFUSE"},
        )
        self.assertEqual(tuple(c.case_id for c in report.cases), ("a", "z"))

    def test_nonblank_string_ids_required(self):
        with self.assertRaises(ValueError):
            compare_cases({"": "ACCEPT"}, {"": "ACCEPT"})
        with self.assertRaises(ValueError):
            compare_cases({1: "ACCEPT"}, {1: "ACCEPT"})

    def test_summary_preserves_review(self):
        report = compare_cases(
            {"x": "ACCEPT", "y": "REFUSE"}, {"x": "REFUSE", "y": "REFUSE"}
        )
        summary = report.to_dict()
        self.assertEqual(summary["gate"], "REVIEW")
        self.assertEqual(summary["new_refusal_needs_review"], 1)
        self.assertEqual(summary["unchanged_refused"], 1)


if __name__ == "__main__":
    unittest.main()