import unittest

import numpy as np
import torch

from src.protocol import (
    ClassCountConditionalCalibrator,
    draw_class_count_window,
    permutation_test_cmmd,
)


class ConditionalCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.embeddings = torch.arange(120, dtype=torch.float32).reshape(60, 2)
        self.labels = torch.tensor([0] * 30 + [1] * 30)

    def test_draw_preserves_requested_class_counts(self):
        emb, labels = draw_class_count_window(
            self.embeddings, self.labels, {0: 12, 1: 15}, np.random.RandomState(1)
        )
        self.assertEqual(emb.shape[0], 27)
        self.assertEqual(int((labels == 0).sum()), 12)
        self.assertEqual(int((labels == 1).sum()), 15)

    def test_threshold_is_cached_by_class_count_signature(self):
        statistic = lambda _ctx, _emb, labels: float((labels == 0).sum())
        calibrator = ClassCountConditionalCalibrator(
            statistic,
            None,
            self.embeddings,
            self.labels,
            [0, 1],
            n_null=20,
            seed=2,
        )
        threshold_a = calibrator.threshold({0: 12, 1: 12})
        threshold_b = calibrator.threshold({0: 15, 1: 12})
        self.assertEqual(threshold_a, 12.0)
        self.assertEqual(threshold_b, 15.0)
        self.assertEqual(len(calibrator._thresholds), 2)

    def test_rejects_class_counts_below_supported_minimum(self):
        calibrator = ClassCountConditionalCalibrator(
            lambda _ctx, _emb, _labels: 0.0,
            None,
            self.embeddings,
            self.labels,
            [0, 1],
            n_null=5,
        )
        with self.assertRaisesRegex(ValueError, 'at least 10 samples per class'):
            calibrator.threshold({0: 9, 1: 11})

    def test_permutation_test_uses_pooled_bandwidth_and_returns_valid_p(self):
        refs = {0: self.embeddings[:15], 1: self.embeddings[30:45]}
        window = torch.cat([self.embeddings[15:27], self.embeddings[45:57]])
        labels = torch.tensor([0] * 12 + [1] * 12)
        statistic, p_value = permutation_test_cmmd(
            refs, window, labels, n_perm=19, seed=3
        )
        self.assertTrue(np.isfinite(statistic))
        self.assertGreaterEqual(p_value, 1 / 20)
        self.assertLessEqual(p_value, 1.0)


if __name__ == '__main__':
    unittest.main()
