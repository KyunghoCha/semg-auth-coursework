"""Recompute historical metrics and check evidence; never load signals or fit models."""
import csv
import hashlib
import io
import json
from pathlib import Path
import numpy as np
if __package__:
    from .features import LABELS
    from .reproduce import EVIDENCE, ROOT, load_metadata, metrics
else:
    from features import LABELS
    from reproduce import EVIDENCE, ROOT, load_metadata, metrics


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)


def read_predictions(path, manifest):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_path = {row["path"]: row for row in manifest}
    y = np.array([LABELS.index(row["true"]) for row in rows])
    p = np.array([[float(row["prob_" + c]) for c in LABELS] for row in rows])
    groups = np.array([int(by_path[row["path"]]["index"]) for row in rows])
    windows = np.array([int(row["window"]) for row in rows])
    assert np.isfinite(p).all() and np.all(p >= 0)
    close(p.sum(1), np.ones(len(p)))
    assert [LABELS[i] for i in p.argmax(1)] == [r["predicted"] for r in rows]
    assert len(set(zip(groups, windows))) == len(rows)
    if "trial_index" in rows[0]:
        assert [int(row["trial_index"]) for row in rows] == groups.tolist()
    for group in dict.fromkeys(groups):
        ix = np.flatnonzero(groups == group)
        assert windows[ix].tolist() == list(range(19))
        assert len(set(y[ix])) == 1
        assert LABELS[y[ix[0]]] == manifest[group]["subject"]
    return rows, y, p, groups


def verify_checksums():
    checked = 0
    for line in (ROOT / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        path = ROOT / name
        assert path.resolve().is_relative_to(ROOT)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
        checked += 1
    return checked


def audit():
    checked = verify_checksums()
    manifest, splits = load_metadata()
    provenance = json.loads((EVIDENCE / "provenance.json").read_text())
    for name, source in provenance["source_artifacts"].items():
        if source["copy"] == "byte-identical":
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == source["source_sha256"], name
    development = splits["holdout"]["train"]
    holdout = splits["holdout"]["test"]
    assert len(development) == 199 and len(holdout) == 50
    assert not set(development) & set(holdout)
    assert set(development + holdout) == set(range(249))
    assert len({row["signal_sha256"] for row in manifest}) == 249
    final = json.loads((EVIDENCE / "final_metrics.json").read_text())
    _, y, p, groups = read_predictions(EVIDENCE / "final_predictions.csv", manifest)
    assert list(dict.fromkeys(groups)) == holdout
    current = metrics(y, p)
    reference = final["calibrated_SVM"]
    published = {"accuracy": 0.9336842105263158, "precision_macro": 0.9345149792021378,
                 "recall_macro": 0.9336842105263157, "f1_macro": 0.933980760427341}
    for key, value in published.items():
        close(current[key], value)
    assert reference["feature_columns"] == 39
    assert reference["support_vectors"] == 1440
    assert reference["serialized_model_bytes"] == 642444
    close(reference["model_only_inference_ms_per_window"], 0.8177362200149219)
    for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "confusion_matrix"):
        close(current[key], reference[key])
    close(current["log_loss"], reference["cross_entropy"])
    assert current["correct"] == 887 and current["n"] == 950
    trial = metrics(y.reshape(-1, 19)[:, 0], p.reshape(-1, 19, 5).mean(1))
    for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
        close(trial[key], reference["trial_mean_probability"][key])
    close(trial["log_loss"], reference["trial_mean_probability"]["cross_entropy"])
    assert hashlib.sha256((EVIDENCE / "final_predictions.csv").read_bytes()).hexdigest() == final["prediction_sha256"]
    completion = json.loads((EVIDENCE / "completion.json").read_text())
    frozen = json.loads((EVIDENCE / "frozen_classical_plan.json").read_text())
    assert frozen["outer_development_folds"] == splits["cv"]
    assert frozen["full_development_indices"] == development
    assert hashlib.sha256((EVIDENCE / "frozen_classical_plan.json").read_bytes()).hexdigest() == completion["plan_sha256"]
    for key, file in (("base_plan_sha256", "predeclared_plan.json"), ("calibration_plan_sha256", "fixed_calibration_plan.json")):
        assert hashlib.sha256((EVIDENCE / file).read_bytes()).hexdigest() == frozen[key]
    plan = json.loads((EVIDENCE / "predeclared_plan.json").read_text())
    for name in ("run_spectral.py", "calibrate_svm.py", "finalize_classical.py"):
        assert hashlib.sha256((EVIDENCE / "historical_source" / name).read_bytes()).hexdigest() == plan["code_sha256"][name]
    rows, oy, op, og = read_predictions(EVIDENCE / "development_oof_predictions.csv", manifest)
    assert set(og) == set(development) and not set(og) & set(holdout)
    fold_ids = np.array([int(row["fold"]) for row in rows])
    # Reconstruct each original CSV, proving the merged OOF table retained its bytes.
    oof_origins = provenance["source_artifacts"]["evidence/development_oof_predictions.csv"]["sources"]
    for source in oof_origins:
        originals = [{k: v for k, v in row.items() if k != "fold"} for row in rows if int(row["fold"]) == source["fold"]]
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=list(originals[0]))
        writer.writeheader()
        writer.writerows(originals)
        assert len(originals) == source["rows"]
        assert hashlib.sha256(stream.getvalue().encode()).hexdigest() == source["source_sha256"]
    fold_metrics = []
    for outer, expected in zip(splits["cv"], completion["folds"]):
        selected = fold_ids == outer["fold"]
        assert list(dict.fromkeys(og[selected])) == outer["test"]
        assert not set(outer["train"]) & set(outer["test"])
        assert set(outer["train"] + outer["test"]) == set(development)
        metric = metrics(oy[selected], op[selected])
        for key in ("accuracy", "f1_macro", "log_loss", "correct", "n", "confusion_matrix"):
            close(metric[key], expected["window"][key])
        fold_metrics.append(metric)
    summary = {}
    for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "log_loss"):
        values = np.array([fold[key] for fold in fold_metrics])
        summary[key] = {"fold_values": values.tolist(), "equal_fold_mean": float(values.mean()),
                        "population_sd_ddof0": float(values.std(ddof=0)), "sample_sd_ddof1": float(values.std(ddof=1))}
        if key in completion["cv_summary"]:
            for stat, value in summary[key].items():
                close(value, completion["cv_summary"][key][stat])
    assert [fold["correct"] for fold in fold_metrics] == [713, 683, 707, 707, 677]
    assert [fold["n"] for fold in fold_metrics] == [760, 760, 760, 760, 741]
    close([fold["f1_macro"] for fold in fold_metrics], [0.9381152486699154, 0.8988548071669127,
          0.9303036555391776, 0.9300871054098986, 0.9111517978387811])
    close(completion["full_development_fit"]["fit_seconds"], 0.7511132750000797)
    assert completion["full_development_fit"]["full_fit_support_vectors"] == 1440
    assert completion["full_model_bytes"] == 642444
    pooled = metrics(oy, op)
    for key, value in completion["pooled_oof_window"].items():
        close(pooled[key], value)
    return {"status": "passed", "scope": "Saved evidence and metric audit only; no raw signals, model loading, fitting, or new test evaluation",
            "checksum_files_verified": checked, "historical_final_window": current,
            "historical_trial_mean_probability": trial, "historical_development_cv": summary,
            "historical_development_pooled": pooled,
            "historical_model": {"features": 39, "support_vectors": 1440, "serialized_bytes": 642444,
              "full_development_fit_seconds": completion["full_development_fit"]["fit_seconds"],
              "model_only_inference_ms_per_window": reference["model_only_inference_ms_per_window"],
              "timing_scope": "Historical cached-feature fitting and precomputed-feature inference; not current measurements"}}


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
