import unittest

from validation_patterns.llm_benchmark import InferenceSample, evaluate_llm


class LLMAnalysisTests(unittest.TestCase):
    def case(self, cid='a', generated=11, ttft=0.1, total=1.1, finish='stop'):
        return InferenceSample(cid, 20, generated, ttft, total, finish)

    def test_decode_only_tokens_per_second(self):
        row = self.case()
        self.assertAlmostEqual(row.decode_tokens_per_second, 10)

    def test_single_token_has_no_decode_speed(self):
        self.assertIsNone(self.case(generated=1).decode_tokens_per_second)

    def test_complete_run(self):
        r = evaluate_llm([self.case('a'), self.case('b', total=2.1)])
        self.assertEqual(r.cases, 2)
        self.assertEqual(r.completed, 2)
        self.assertEqual(r.completion_rate, 1)
        self.assertAlmostEqual(r.median_total_seconds, 1.6)
        self.assertEqual(r.truncated, 0)

    def test_truncated_is_not_complete(self):
        r = evaluate_llm([self.case('a'), self.case('b', finish='length')])
        self.assertEqual(r.completion_rate, 0.5)
        self.assertEqual(r.truncated, 1)
        self.assertEqual(r.throughput_samples, 1)

    def test_errors_are_not_completed(self):
        r = evaluate_llm([self.case('a', finish='error')])
        self.assertEqual(r.errors, 1)
        self.assertIsNone(r.median_total_seconds)
        self.assertIsNone(r.median_decode_tokens_per_second)

    def test_no_samples_is_invalid(self):
        with self.assertRaises(ValueError):
            evaluate_llm([])

    def test_duplicate_cases_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_llm([self.case('dup'), self.case('dup')])

    def test_rejects_negative_token_count(self):
        with self.assertRaises(ValueError):
            InferenceSample('a', -1, 2, 0.1, 1, 'stop')

    def test_rejects_zero_generated_tokens(self):
        with self.assertRaises(ValueError):
            self.case(generated=0)

    def test_rejects_invalid_finish_reason(self):
        with self.assertRaises(ValueError):
            self.case(finish='maybe')

    def test_rejects_invalid_timing(self):
        with self.assertRaises(ValueError):
            self.case(ttft=2, total=1)

    def test_rejects_boolean_token_count(self):
        with self.assertRaises(ValueError):
            self.case(generated=True)

    def test_single_token_completion_is_not_tps_data(self):
        r = evaluate_llm([self.case(generated=1)])
        self.assertEqual(r.completed, 1)
        self.assertEqual(r.throughput_samples, 0)
        self.assertIsNone(r.median_decode_tokens_per_second)


if __name__ == '__main__':
    unittest.main()
