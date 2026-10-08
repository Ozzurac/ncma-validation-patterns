import unittest

from validation_patterns.asr_benchmark import (
    SpeechSample, evaluate_asr, normalize_transcript,
)


class ASRBenchmarkTests(unittest.TestCase):
    def sample(self, cid='a', ref='Olá, MUNDO!', hyp='olá mundo', dur=4.0, inf=0.1):
        return SpeechSample(cid, ref, hyp, dur, inf)

    def test_normalization_is_explicit_and_stable(self):
        self.assertEqual(normalize_transcript('  OLÁ,  Mundo!! '), 'olá mundo')

    def test_punctuation_is_not_counted_as_a_word(self):
        self.assertEqual(normalize_transcript('hello-world'), 'hello world')

    def test_accent_is_preserved(self):
        self.assertNotEqual(normalize_transcript('maçã'), normalize_transcript('maca'))

    def test_perfect_score(self):
        r = evaluate_asr([self.sample()])
        self.assertEqual(r.wer, 0)
        self.assertEqual(r.cer, 0)
        self.assertEqual(r.exact_clips, 1)
        self.assertAlmostEqual(r.realtime_factor, 0.025)
        self.assertIsNone(r.warm_median_seconds)

    def test_one_substitution(self):
        r = evaluate_asr([self.sample(ref='one two', hyp='one three')])
        self.assertEqual(r.word_errors, 1)
        self.assertAlmostEqual(r.wer, 0.5)

    def test_one_insertion(self):
        r = evaluate_asr([self.sample(ref='a b', hyp='a b c')])
        self.assertAlmostEqual(r.wer, 0.5)

    def test_one_deletion(self):
        r = evaluate_asr([self.sample(ref='a b', hyp='a')])
        self.assertAlmostEqual(r.wer, 0.5)

    def test_empty_hypothesis_is_a_valid_miss(self):
        r = evaluate_asr([self.sample(ref='a b', hyp='')])
        self.assertEqual(r.wer, 1)

    def test_micro_averaging_uses_reference_token_weights(self):
        rows = [self.sample('a', 'a', ''), self.sample('b', 'a b c', 'a b c')]
        r = evaluate_asr(rows)
        self.assertEqual(r.reference_words, 4)
        self.assertAlmostEqual(r.wer, 0.25)
        self.assertEqual(r.exact_clips, 1)

    def test_first_clip_is_not_in_warm_median(self):
        rows = [self.sample('a', dur=4, inf=2), self.sample('b', dur=4, inf=0.2),
                self.sample('c', dur=4, inf=0.1)]
        r = evaluate_asr(rows)
        self.assertAlmostEqual(r.warm_median_seconds, 0.15)
        self.assertEqual(r.p95_inference_seconds, 2)
        self.assertAlmostEqual(r.realtime_factor, 2.3/12)

    def test_rejects_empty_corpus(self):
        with self.assertRaises(ValueError):
            evaluate_asr([])

    def test_rejects_duplicate_clip_ids(self):
        with self.assertRaises(ValueError):
            evaluate_asr([self.sample('dup'), self.sample('dup')])

    def test_rejects_empty_ground_truth(self):
        with self.assertRaises(ValueError):
            self.sample(ref='??')

    def test_rejects_zero_duration(self):
        with self.assertRaises(ValueError):
            self.sample(dur=0)

    def test_rejects_nan_inference(self):
        with self.assertRaises(ValueError):
            self.sample(inf=float('nan'))

    def test_rejects_non_string_transcript(self):
        with self.assertRaises(ValueError):
            SpeechSample('a', 'hello', None, 1, 1)


if __name__ == '__main__':
    unittest.main()
