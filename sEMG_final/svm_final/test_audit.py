"""CPU-only evidence and synthetic-formula checks. No fitting or raw dataset reads."""
import ast
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, periodogram
from sklearn.model_selection import StratifiedKFold
import audit
import features
import model
from reproduce import EVIDENCE, load_metadata


def historical_functions(filename, names, namespace):
    tree = ast.parse((EVIDENCE / "historical_source" / filename).read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    code = compile(ast.Module(body=selected, type_ignores=[]), str(filename), "exec")
    exec(code, namespace)
    return namespace


class TestFrozenSVM(unittest.TestCase):
    def test_all_historical_evidence(self):
        with patch("sklearn.svm.SVC.fit", side_effect=AssertionError("Audit must not fit a model")):
            result = audit.audit()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["historical_final_window"]["correct"], 887)
        self.assertEqual(result["historical_development_pooled"]["n"], 3781)

    def test_feature_formula_ast_unchanged(self):
        source = ast.parse((EVIDENCE / "historical_source/run_spectral.py").read_text())
        packaged = ast.parse(Path(features.__file__).read_text())
        for name in ("feature_window", "transform_trial"):
            old = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == name)
            new = next(node for node in packaged.body if isinstance(node, ast.FunctionDef) and node.name == name)
            self.assertEqual(ast.dump(old), ast.dump(new))

    def test_feature_order_and_dimension(self):
        plan = json.loads((EVIDENCE / "predeclared_plan.json").read_text())
        self.assertEqual(features.AMP_NAMES + features.SHAPE_NAMES, plan["features"]["combined"])
        self.assertEqual(len(features.AMP_NAMES), 4)
        self.assertEqual(len(features.SHAPE_NAMES), 35)

    def test_synthetic_transform_exact_parity(self):
        old = historical_functions("run_spectral.py", {"feature_window", "transform_trial"},
            {"np": np, "butter": butter, "filtfilt": filtfilt, "iirnotch": iirnotch,
             "periodogram": periodogram, "BANDS": features.BANDS, "WIN": 300, "HOP": 150})
        rng = np.random.default_rng(2026)
        for amplitude in (0.02, 1.0, 10.0):
            signal = rng.normal(size=(3000, 2)) * amplitude + np.array([2., -3.])
            actual = features.transform_trial(signal)
            self.assertEqual(actual.shape, (19, 39))
            np.testing.assert_array_equal(actual, old["transform_trial"](signal))

    def test_invalid_signal_is_rejected(self):
        for signal in (np.ones((2999, 2)), np.ones((3000, 3)), np.full((3000, 2), np.nan)):
            with self.assertRaises(AssertionError):
                features.transform_trial(signal)

    def test_exact_full_development_calibration_folds(self):
        manifest, splits = load_metadata()
        ordered = splits["holdout"]["train"]
        groups = np.repeat(ordered, 19)
        y = np.repeat([features.LABELS.index(manifest[i]["subject"]) for i in ordered], 19)
        _, actual = model.grouped_calibration_folds(y, groups)
        completion = json.loads((EVIDENCE / "completion.json").read_text())
        self.assertEqual(actual, completion["full_development_fit"]["calibration_folds"])

    def test_exact_outer_training_calibration_folds(self):
        manifest, splits = load_metadata()
        completion = json.loads((EVIDENCE / "completion.json").read_text())
        for outer, recorded in zip(splits["cv"], completion["folds"]):
            groups = np.repeat(outer["train"], 19)
            y = np.repeat([features.LABELS.index(manifest[i]["subject"]) for i in outer["train"]], 19)
            folds, provenance = model.grouped_calibration_folds(y, groups)
            self.assertEqual(provenance, recorded["fit"]["calibration_folds"])
            seen = np.zeros(len(y), dtype=int)
            for fit, calibration in folds:
                self.assertFalse(set(groups[fit]) & set(groups[calibration]))
                self.assertFalse(set(groups[fit]) & set(outer["test"]))
                self.assertEqual(set(fit) | set(calibration), set(range(len(y))))
                seen[calibration] += 1
            np.testing.assert_array_equal(seen, 1)

    def test_fixed_estimator_settings_without_fitting(self):
        groups = np.repeat(np.arange(30), 19)
        y = np.repeat(np.repeat(np.arange(5), 6), 19)
        estimator, _ = model.build_model(y, groups)
        self.assertEqual(estimator.method, "sigmoid")
        self.assertIs(estimator.ensemble, False)
        self.assertEqual(estimator.n_jobs, 1)
        self.assertEqual(len(estimator.cv), 3)
        pipe = estimator.estimator
        self.assertEqual(type(pipe[0]).__name__, "StandardScaler")
        self.assertFalse(hasattr(pipe[0], "mean_"))
        svc = pipe[1]
        for name, value in {"C": 1.0, "kernel": "rbf", "gamma": "scale", "probability": False,
                            "cache_size": 256, "tol": 1e-3, "decision_function_shape": "ovr", "class_weight": None}.items():
            self.assertEqual(getattr(svc, name), value)

    def test_original_group_split_formula_parity(self):
        old = historical_functions("calibrate_svm.py", {"grouped_calibration_folds"},
                                   {"np": np, "StratifiedKFold": StratifiedKFold})
        groups = np.repeat(np.arange(30)[::-1], 19)
        y = np.repeat(np.repeat(np.arange(5), 6), 19)
        current_folds, current = model.grouped_calibration_folds(y, groups)
        previous_folds, previous = old["grouped_calibration_folds"](y, groups)
        self.assertEqual(current, previous)
        for actual, expected in zip(current_folds, previous_folds):
            for a, b in zip(actual, expected):
                np.testing.assert_array_equal(a, b)

    def test_no_bundled_raw_data_features_or_models(self):
        root = Path(__file__).resolve().parent
        for file in root.rglob("*"):
            if file.is_file() and "__pycache__" not in file.parts:
                self.assertNotIn(file.suffix, (".joblib", ".npz", ".npy", ".pth", ".pt"))
        self.assertFalse((root / "data").exists())


if __name__ == "__main__":
    unittest.main()
