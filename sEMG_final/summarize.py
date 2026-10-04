"""Recompute stored predictions, then summarize the prespecified experiments."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from data_utils import LABELS

ROOT = Path(__file__).resolve().parent
MODELS = ["SimpleCNN", "ResNet18", "DenseNet161"]
METRICS = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]


def load_verified_run(path):
    data = json.loads((path / "metrics.json").read_text())
    with (path / "predictions.csv").open(newline="") as stream:
        predictions = list(csv.DictReader(stream))
    true = [row["true"] for row in predictions]
    pred = [row["predicted"] for row in predictions]
    if len(true) != data["n_test_windows"]:
        raise ValueError(f"Prediction count mismatch: {path}")
    cm = confusion_matrix(true, pred, labels=LABELS)
    precision, recall, f1, _ = precision_recall_fscore_support(true, pred, labels=LABELS, average="macro", zero_division=0)
    recomputed = dict(zip(METRICS, [accuracy_score(true, pred), precision, recall, f1]))
    for name, value in recomputed.items():
        if not np.isclose(data[name], value, rtol=0, atol=1e-12):
            raise ValueError(f"Metric mismatch: {path}/{name}")
    if cm.tolist() != data["confusion_matrix"]:
        raise ValueError(f"Confusion matrix mismatch: {path}")
    if len({(row["path"], row["window"]) for row in predictions}) != len(predictions):
        raise ValueError(f"Duplicate trial/window predictions: {path}")
    return data, predictions


def draw_confusion(cm, title, destination):
    fig, ax = plt.subplots(figsize=(4.2, 3.7))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=190)
    ax.set_xticks(range(5), LABELS); ax.set_yticks(range(5), LABELS)
    ax.set(xlabel="Predicted", ylabel="True")
    ax.set_title(title + "\nRed outline: maximum error", fontsize=10)
    off = cm.copy(); np.fill_diagonal(off, 0)
    maximum = int(off.max())
    if maximum:
        for i, j in zip(*np.where(off == maximum)):
            if i != j:
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, edgecolor="red", linewidth=1.8))
    for i in range(5):
        for j in range(5):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > 95 else "black", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=.8)
    fig.tight_layout(); fig.savefig(destination, dpi=160); plt.close(fig)


def summarize():
    results = ROOT / "results"
    protocol = json.loads((results / "protocol.json").read_text())
    all_run_rows, comparison, error_details = [], [], []
    for name in MODELS:
        runs = []
        for seed in [42, 43, 44]:
            path = results / "runs" / f"holdout_{name}_seed{seed}"
            run, predictions = load_verified_run(path)
            if run["config"]["epochs"] != protocol["epochs"]:
                raise ValueError("Unequal epoch budget")
            if run["n_test_windows"] != 950:
                raise ValueError("The expected final test set has 950 windows")
            runs.append(run)
            all_run_rows.append({"phase": "holdout", "model": name, "repeat_or_fold": seed,
                                 **{metric: run[metric] for metric in METRICS}, "train_seconds": run["train_seconds"]})
            if seed == 42:
                cm = np.array(run["confusion_matrix"])
                if not np.array_equal(cm.sum(axis=1), np.full(5, 190)):
                    raise ValueError("Expected 190 test windows per true identity")
                draw_confusion(cm, f"{name}, seed 42", results / f"confusion_{name}.png")
                off = cm.copy(); np.fill_diagonal(off, 0)
                maximum = int(off.max())
                pairs = [(LABELS[i], LABELS[j]) for i, j in zip(*np.where(off == maximum)) if i != j] if maximum else []
                examples = []
                for true_label, predicted_label in pairs:
                    row = next(row for row in predictions if row["true"] == true_label and row["predicted"] == predicted_label)
                    examples.append(row)
                with (path / "training.csv").open(newline="") as stream:
                    training = list(csv.DictReader(stream))
                error_details.append({"model": name, "seed": 42, "maximum_error_count": maximum,
                                      "maximum_error_pairs": pairs, "examples": examples,
                                      "reverse_error_count": int(cm[LABELS.index(pairs[0][1]), LABELS.index(pairs[0][0])]) if pairs else 0,
                                      "destination_predictions": int(cm[:, LABELS.index(pairs[0][1])].sum()) if pairs else 0,
                                      "destination_support": int(cm[LABELS.index(pairs[0][1]), :].sum()) if pairs else 0,
                                      "final_epoch_online_training_accuracy": float(training[-1]["train_accuracy"]),
                                      "test_accuracy": run["accuracy"]})
        item = {"model": name}
        for metric in METRICS:
            values = [run[metric] for run in runs]
            item[metric + "_mean"] = float(np.mean(values))
            item[metric + "_sd"] = float(np.std(values, ddof=0))
        item["train_seconds_mean"] = float(np.mean([run["train_seconds"] for run in runs]))
        item["inference_ms_mean"] = float(np.mean([run["inference_ms_per_window"] for run in runs]))
        item["parameters"] = runs[0]["parameters"]
        comparison.append(item)
    folds = []
    for fold in range(1, 6):
        run, _ = load_verified_run(results / "runs" / f"cv_DenseNet161_fold{fold}")
        row = {"fold": fold, **{metric: run[metric] for metric in METRICS},
               "n_validation_trials": run["n_test_trials"], "n_validation_windows": run["n_test_windows"]}
        folds.append(row)
        all_run_rows.append({"phase": "cv", "model": "DenseNet161", "repeat_or_fold": fold,
                             **{metric: run[metric] for metric in METRICS}, "train_seconds": run["train_seconds"]})
    cv = {metric + suffix: float(function([fold[metric] for fold in folds]))
          for metric in METRICS for suffix, function in [("_mean", np.mean), ("_sd", np.std)]}
    ranking = sorted(comparison, key=lambda row: (-row["accuracy_mean"], row["train_seconds_mean"]))
    summary = {"comparison": comparison, "best_model": ranking[0]["model"], "worst_model": ranking[-1]["model"],
               "ranking_rule": "mean holdout accuracy over three fixed training seeds; ties by shorter mean training time",
               "standard_deviation": "population SD (ddof=0)", "errors_seed42": error_details,
               "cv_DenseNet161": {"scope": "199 holdout training trials only", "folds": folds, **cv},
               "verified_prediction_runs": len(all_run_rows), "metric_unit": "300ms window"}
    for filename, rows in [("comparison.csv", comparison), ("cross_validation.csv", folds), ("all_runs.csv", all_run_rows)]:
        with (results / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (results / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(summarize(), ensure_ascii=False, indent=2))
