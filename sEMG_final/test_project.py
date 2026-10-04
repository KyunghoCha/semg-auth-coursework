"""Small regression checks; no model training or test-set tuning."""
import unittest
import numpy as np
import torch

from data_utils import load_trials
from experiment import ROOT, LABELS, make_plan, make_windows, normalize_windows, to_cwt, model_for


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trials, cls.duplicates = load_trials(ROOT / "data" / "data")
        cls.plan = make_plan(cls.trials)
        torch.set_num_threads(2)

    def test_source_quality_and_duplicate(self):
        self.assertEqual(len(self.trials), 249)
        self.assertEqual(self.duplicates, [{"removed": "E/e (50).csv", "kept": "E/e (38).csv"}])

    def test_holdout_and_cv_are_trial_disjoint(self):
        train, test = (set(self.plan["holdout"][k]) for k in ["train", "test"])
        self.assertEqual((len(train), len(test)), (199, 50))
        self.assertFalse(train & test)
        self.assertEqual({label: sum(self.trials[i].subject == label for i in test) for label in LABELS}, dict.fromkeys(LABELS, 10))
        val_indices = []
        for fold in self.plan["cv"]:
            tr, va = set(fold["train"]), set(fold["test"])
            self.assertFalse(tr & va); self.assertFalse((tr | va) & test)
            self.assertEqual(tr | va, train)
            self.assertFalse({self.trials[i].signal_sha256 for i in tr} & {self.trials[i].signal_sha256 for i in va})
            val_indices.extend(va)
        self.assertEqual(set(val_indices), train)
        self.assertEqual(len(val_indices), len(train))

    def test_window_normalization_and_cwt(self):
        windows = make_windows(self.trials[0].signal)
        self.assertEqual(windows.shape, (19, 300, 2))
        normal = normalize_windows(windows)
        self.assertGreaterEqual(normal.min(), 0); self.assertLessEqual(normal.max(), 1)
        constant = normalize_windows(np.ones((2, 300, 2)))
        self.assertTrue(np.isfinite(constant).all()); self.assertTrue((constant == 0).all())
        cwt = to_cwt(normal[0]); self.assertEqual(cwt.shape, (3, 32, 300))
        self.assertTrue(np.isfinite(cwt).all())
        np.testing.assert_allclose(cwt[2], (cwt[0] + cwt[1]) / 2, rtol=2e-7, atol=1e-7)

    def test_three_fresh_model_outputs(self):
        for name in ["SimpleCNN", "ResNet18", "DenseNet161"]:
            model = model_for(name).eval()
            with torch.inference_mode():
                self.assertEqual(tuple(model(torch.zeros(2, 3, 32, 300)).shape), (2, 5))


if __name__ == "__main__":
    unittest.main()
