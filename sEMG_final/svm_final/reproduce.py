"""Optional fixed-recipe rerun; never runs a search or overwrites saved evidence."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

# Set before NumPy/scikit-learn imports, independent of available CPU IDs.
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, precision_score, recall_score
if __package__:
    from .features import LABELS, transform_trial
    from .model import fit_fixed, predict_probabilities
else:
    from features import LABELS, transform_trial
    from model import fit_fixed, predict_probabilities

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def load_metadata():
    with (EVIDENCE / "manifest.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    splits = json.loads((EVIDENCE / "splits.json").read_text())
    original = json.loads((EVIDENCE / "predeclared_plan.json").read_text())
    for name, expected in original["source_sha256"].items():
        assert digest(EVIDENCE / name) == expected
    assert len(rows) == 249 and [int(r["index"]) for r in rows] == list(range(249))
    return rows, splits


def check_versions():
    versions = json.loads((EVIDENCE / "predeclared_plan.json").read_text())["versions"]
    for package, expected in versions.items():
        observed = importlib.metadata.version(package)
        if observed != expected:
            raise RuntimeError(f"Expected {package}=={expected}, found {observed}")
    return versions


def extract(data_dir, indices, manifest):
    """Read only requested whole trials; check file bytes and decoded signal."""
    root = Path(data_dir).resolve()
    result = {key: [] for key in ("X", "y", "groups", "windows")}
    for index in indices:
        row = manifest[index]
        path = (root / row["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Manifest path escapes data directory")
        if digest(path) != row["sha256"]:
            raise ValueError(f"File hash mismatch: {row['path']}")
        signal = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
        if hashlib.sha256(signal.astype("<f8").tobytes()).hexdigest() != row["signal_sha256"]:
            raise ValueError(f"Signal hash mismatch: {row['path']}")
        features = transform_trial(signal)
        result["X"].extend(features)
        result["y"].extend([LABELS.index(row["subject"])] * 19)
        result["groups"].extend([index] * 19)
        result["windows"].extend(range(19))
    result = {key: np.asarray(value) for key, value in result.items()}
    assert result["X"].shape == (19 * len(indices), 39)
    return result


def subset(ds, ordered_indices):
    positions = np.concatenate([np.flatnonzero(ds["groups"] == i) for i in ordered_indices])
    assert len(positions) == len(ordered_indices) * 19
    return {key: value[positions] for key, value in ds.items()}


def metrics(y, p):
    pred = p.argmax(1)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision_macro": float(precision_score(y, pred, labels=range(5), average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, pred, labels=range(5), average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y, pred, labels=range(5), average="macro", zero_division=0)),
        "log_loss": float(log_loss(y, p, labels=range(5))),
        "correct": int(np.sum(y == pred)), "n": len(y),
        "confusion_matrix": confusion_matrix(y, pred, labels=range(5)).tolist(),
    }


def save_predictions(path, ds, p, manifest):
    with Path(path).open("w", newline="") as f:
        fields = ["path", "trial_index", "window", "true", "predicted"] + ["prob_" + label for label in LABELS]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, (group, window, y) in enumerate(zip(ds["groups"], ds["windows"], ds["y"])):
            writer.writerow({"path": manifest[int(group)]["path"], "trial_index": int(group),
                "window": int(window), "true": LABELS[int(y)], "predicted": LABELS[int(p[i].argmax())],
                **{"prob_" + label: float(p[i, k]) for k, label in enumerate(LABELS)}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("cv", "fit", "evaluate"))
    parser.add_argument("--data-dir", required=True, type=Path, help="Directory containing A/ through E/ CSVs")
    parser.add_argument("--output", required=True, type=Path, help="New directory outside bundled evidence")
    parser.add_argument("--model", type=Path, help="Trusted model.joblib created by this package's fit command")
    parser.add_argument("--acknowledge-reused-holdout", action="store_true")
    args = parser.parse_args()
    if args.stage == "evaluate" and (not args.acknowledge_reused_holdout or args.model is None):
        parser.error("evaluate requires --model and --acknowledge-reused-holdout; this is not a clean test")
    output = args.output.resolve()
    if output == ROOT or output.is_relative_to(EVIDENCE):
        parser.error("Refusing to write into source/evidence")
    if output.exists():
        parser.error("Output already exists; choose a new directory")
    versions = check_versions()
    manifest, splits = load_metadata()
    output.mkdir(parents=True)
    run = {"stage": args.stage, "versions": versions,
           "scope": "New execution of unchanged recipe, historical reused split; not independent validation",
           "manifest_sha256": digest(EVIDENCE / "manifest.csv"),
           "splits_sha256": digest(EVIDENCE / "splits.json"),
           "code_sha256": {name: digest(ROOT / name) for name in ("features.py", "model.py", "reproduce.py")}}
    if args.stage == "cv":
        development = extract(args.data_dir, splits["holdout"]["train"], manifest)
        folds = []
        for outer in splits["cv"]:
            tr, va = subset(development, outer["train"]), subset(development, outer["test"])
            assert not set(tr["groups"]) & set(va["groups"])
            model, fit = fit_fixed(tr["X"], tr["y"], tr["groups"])
            p = predict_probabilities(model, va["X"])
            row = {"fold": outer["fold"], "window": metrics(va["y"], p), "fit": fit}
            folds.append(row)
            save_predictions(output / f"fold{outer['fold']}_predictions.csv", va, p, manifest)
        stats = {}
        for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "log_loss"):
            values = np.array([fold["window"][key] for fold in folds])
            stats[key] = {"fold_values": values.tolist(), "equal_fold_mean": float(values.mean()),
                          "population_sd_ddof0": float(values.std(ddof=0)), "sample_sd_ddof1": float(values.std(ddof=1))}
        run.update(folds=folds, cv_summary=stats,
                   interpretation="Postselection development stability; not nested recipe selection")
    elif args.stage == "fit":
        tr = extract(args.data_dir, splits["holdout"]["train"], manifest)
        model, fit = fit_fixed(tr["X"], tr["y"], tr["groups"])
        joblib.dump(model, output / "model.joblib")
        run.update(fit=fit, training_indices=splits["holdout"]["train"], model_sha256=digest(output / "model.joblib"))
    else:
        # joblib is pickle-based: only load your own trusted output.
        previous = json.loads((args.model.parent / "run.json").read_text())
        assert previous["stage"] == "fit"
        assert previous["training_indices"] == splits["holdout"]["train"]
        for key in ("manifest_sha256", "splits_sha256", "code_sha256"):
            assert previous[key] == run[key]
        assert digest(args.model) == previous["model_sha256"]
        model = joblib.load(args.model)
        test = extract(args.data_dir, splits["holdout"]["test"], manifest)
        p = predict_probabilities(model, test["X"])
        run.update(window=metrics(test["y"], p), trial_mean_probability=metrics(
            test["y"].reshape(-1, 19)[:, 0], p.reshape(-1, 19, 5).mean(1)))
        save_predictions(output / "predictions.csv", test, p, manifest)
    write_json(output / "run.json", run)
    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
