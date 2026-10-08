import unittest

from validation_patterns.split_audit import (
    AuditStatus, DevCheckpoint, Example, Split, audit_splits, select_checkpoint,
)


class DatasetSplitTests(unittest.TestCase):
    def items(self):
        return [Example('train-1', Split.TRAIN, 'group-a', 'Open a file'),
                Example('dev-1', Split.DEV, 'group-b', 'Find an error'),
                Example('hold-1', Split.HOLDOUT, 'group-c', 'Explain a result')]

    def test_healthy_splits_pass_basic_audit(self):
        r = audit_splits(self.items())
        self.assertEqual(r.status, AuditStatus.PASS)
        self.assertEqual(r.records, 3)

    def test_empty_corpus_is_not_a_pass(self):
        self.assertEqual(audit_splits([]).status, AuditStatus.INCONCLUSIVE)

    def test_missing_holdout_cannot_pass(self):
        self.assertEqual(audit_splits(self.items()[:2]).status, AuditStatus.INCONCLUSIVE)

    def test_exact_prompt_contamination(self):
        rows = self.items()
        rows[-1] = Example('hold-1', Split.HOLDOUT, 'group-c', '  OPEN   A  FILE ')
        r = audit_splits(rows)
        self.assertEqual(r.status, AuditStatus.FAIL)
        self.assertEqual(r.cross_split_prompts, 1)

    def test_group_based_leakage(self):
        rows = self.items()
        rows[-1] = Example('hold-1', Split.HOLDOUT, 'group-a', 'Different words')
        r = audit_splits(rows)
        self.assertEqual(r.status, AuditStatus.FAIL)
        self.assertEqual(r.cross_split_groups, 1)

    def test_same_group_within_same_split_is_not_cross_split(self):
        rows = self.items() + [Example('train-2', Split.TRAIN, 'group-a', 'A new prompt')]
        self.assertEqual(audit_splits(rows).status, AuditStatus.PASS)

    def test_duplicate_ids_fail(self):
        rows = self.items() + [Example('train-1', Split.TRAIN, 'group-d', 'Different')]
        self.assertEqual(audit_splits(rows).duplicate_ids, 1)
        self.assertEqual(audit_splits(rows).status, AuditStatus.FAIL)

    def test_blank_groups_disallowed(self):
        with self.assertRaises(ValueError):
            Example('id', Split.TRAIN, ' ', 'Prompt')

    def test_unknown_split_disallowed(self):
        with self.assertRaises(ValueError):
            Example('id', 'TEST', 'group', 'Prompt')

    def test_later_epoch_training_loss_is_not_a_selection_criterion(self):
        rows = [DevCheckpoint(1, 36, 40, 0.09, 0.03),
                DevCheckpoint(2, 40, 40, 0.03, 0.03),
                DevCheckpoint(3, 40, 40, 0.03, 0.03)]
        self.assertEqual(select_checkpoint(rows, max_general_wer=0.05).epoch, 2)

    def test_good_dev_is_rejected_on_general_regression(self):
        rows = [DevCheckpoint(1, 38, 40, 0.05, 0.02),
                DevCheckpoint(2, 40, 40, 0.0, 0.40)]
        self.assertEqual(select_checkpoint(rows, max_general_wer=0.05).epoch, 1)

    def test_no_eligible_checkpoint_must_fail(self):
        with self.assertRaises(ValueError):
            select_checkpoint([DevCheckpoint(1, 20, 40, 0.5, 0.4)],
                              max_general_wer=0.05)

    def test_duplicate_epochs_must_fail(self):
        with self.assertRaises(ValueError):
            select_checkpoint([DevCheckpoint(1, 30, 40, 0.1, 0.03),
                               DevCheckpoint(1, 39, 40, 0.03, 0.03)],
                              max_general_wer=0.05)

    def test_invalid_dev_metric_must_fail(self):
        with self.assertRaises(ValueError):
            DevCheckpoint(1, 50, 40, 0.1, 0.1)

    def test_nan_generalization_gate_must_fail(self):
        with self.assertRaises(ValueError):
            select_checkpoint([], max_general_wer=float('nan'))


if __name__ == '__main__':
    unittest.main()
